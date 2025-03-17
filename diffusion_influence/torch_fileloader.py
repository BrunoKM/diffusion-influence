from os import PathLike
from typing import Callable, Generic, Iterable, Optional, Sequence, TypeVar

import torch
import torch.multiprocessing as mp

T, U = TypeVar("T"), TypeVar("U")


class TorchFileLoader(Generic[T, U]):
    """A class for loading a sequence of (torch) files in parallel using torch multiprocessing.

    It takes a sequence of paths pointing to files that can be loaded with torch.load,
    and creates an iterable that loads them and puts them on a desired device in
    parallel using a multiprocess queue. This can make loading large files to GPU/CPU memory faster
    by loading them asynchronously.

    It acts fundamentally quite similarly to a pytorch DataLoader, but:
      - allows for putting multiple files on the target device in parallel, whereas
        DataLoader only transfers one data batch at a time
      - allows for controlling things like the queue size

    These are crucial when transferring large files to GPU memory, as the transfer time
    is the bottleneck in the process.

    Note that because the objects are loaded asynchronously, the order in which they are
    returned is not guaranteed to be the same as the order of the input files. Hence,
    the iterable returns a tuple of (index, loaded_object) where index is the index of
    the file in the input sequence of paths.

    Important: it's crucial that the consumer of the iterable doesn't keep references to
    the loaded objects longer than they are needed, as the sending process is
    required to keep the original objects in memory until the consumer no longer
    retains a copy. Hence, after using the laoded objects on the consumer side, calling
    `del` on the objects is recommended to free up memory.
    https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors
    """

    def __init__(
        self,
        files: Sequence[PathLike],
        num_workers: int,
        max_queue_size: int,
        device: torch.device,
        data_map: Optional[Callable[[T], U]] = None,
    ):
        """
        Create an Iterable object that loads files in parallel.

        Args:
            files: A sequence of paths to files that can be loaded with torch.load
            num_workers: The number of worker processes to use for loading the files
                in parallel
            max_queue_size: The maximum size of the queue that holds the loaded files.
                This will be limited by the amount of memory available on the target
                device, since the queue will hold the loaded files in memory on the target
                device.
            device: The target device to put the loaded files on.
            data_map: An optional function that maps the loaded data to another form
                before putting them in the queue. This can be useful if the loaded
                data needs to be transformed in some way before being put on the target
                device.

        """
        self.files = files
        self.num_workers = num_workers
        self.max_queue_size = max_queue_size
        self.data_map = data_map
        self.device = device

    def __iter__(self) -> Iterable[tuple[int, U]]:
        context = mp.get_context("spawn")  # Required for torch
        processing_complete_event = context.Event()

        index_queue: "mp.Queue[tuple[int, PathLike] | None]" = context.Queue(
            maxsize=len(self.files) + self.num_workers
        )

        # Add files to the queue
        for i, file in enumerate(self.files):
            index_queue.put((i, file))

        loaded_obj_queue: "mp.Queue[tuple[int, U]]" = context.Queue(
            maxsize=self.max_queue_size
        )
        processes: list[mp.Process] = []
        try:
            for _ in range(self.num_workers):
                process = context.Process(
                    target=torch_file_loader_worker,
                    args=(
                        index_queue,
                        loaded_obj_queue,
                        self.data_map,
                        self.device,
                        processing_complete_event,
                    ),
                )

                process.start()
                processes += [process]  #  type: ignore

            num_finished = 0
            while num_finished < len(self.files):
                idx, loaded_obj = loaded_obj_queue.get(timeout=5)
                yield idx, loaded_obj
                num_finished += 1
            processing_complete_event.set()
        finally:
            processing_complete_event.set()
            index_queue.cancel_join_thread()
            loaded_obj_queue.cancel_join_thread()
            index_queue.close()
            loaded_obj_queue.close()

            for p in processes:
                p.join(timeout=0.1)  # Ensure all processes have terminated
            for p in processes:
                if p.is_alive():
                    p.terminate()


def torch_file_loader_worker(
    index_queue: "mp.Queue[tuple[int, PathLike] | None]",
    loaded_obj_queue: "mp.Queue[U]",
    data_map: Optional[Callable],
    map_device: torch.device,
    processing_complete_event: mp.Event,
) -> None:
    while index_queue.qsize() > 0:
        result = index_queue.get(timeout=1)
        idx, file = result
        loaded_obj = torch.load(file, map_location=map_device)
        if data_map is not None:
            loaded_obj = data_map(loaded_obj)
        nested_share_memory(loaded_obj)
        loaded_obj_queue.put((idx, loaded_obj))
    processing_complete_event.wait()


def nested_share_memory(o):
    """For a structured object that's a nested list/dictionary/tuple of torch tensors, share memory for all tensors."""
    match o:
        case torch.Tensor():
            o.share_memory_()
        case list() | tuple():
            for item in o:
                nested_share_memory(item)
        case dict():
            for item in o.values():
                nested_share_memory(item)
        case _:
            pass
