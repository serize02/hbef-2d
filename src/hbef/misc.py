import itertools
import sys
import threading
import time
import typing as tp
from types import TracebackType

import psutil

T = tp.TypeVar('T')
@tp.runtime_checkable
class IterableSized(tp.Protocol[T]):
    def __iter__(self) -> tp.Iterator[T]: ...
    def __len__(self) -> int: ...

class cpuspin:
    def __init__(self, iterable: IterableSized[T], desc: str = "Running") -> None:
        self.__iterable = iterable
        self.__desc = desc
        self.__stop_event = threading.Event()
        self.__thread = None
        self.__interval = 0.1
        self.__lock = threading.Lock()
        self.__count = 0
        self.__total = len(self.__iterable)
        psutil.cpu_percent(interval=None)

    @property
    def __pct(self) -> float:
        return (self.__count / self.__total) * 100
    
    @property
    def __cpu_us(self) -> float:
        return psutil.cpu_percent()

    def __animate(self) -> None:
        for c in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
            if self.__stop_event.is_set():
                break
            with self.__lock:
                sys.stdout.write(f'\r{self.__desc} {self.__pct:6.2f}% {c}  CPU {self.__cpu_us:2.0f}%')
                sys.stdout.flush()
            time.sleep(self.__interval)

    def __iter__(self) -> tp.Iterator:
        for item in self.__iterable:
            with self.__lock:
                self.__count += 1
            yield item
        
    def __enter__(self) -> tp.Self:
        self.__thread = threading.Thread(target=self.__animate, daemon=True)
        self.__thread.start()
        return self
    
    def __exit__(self, 
                 exc_type: type[BaseException] | None,
                 exc_value: BaseException | None,
                 traceback: TracebackType | None) -> bool | None:
        self.__stop_event.set()
        if self.__thread is not None:
            self.__thread.join()
        if exc_type is None:
            sys.stdout.write(f'\r{self.__desc} {100.0:6.2f}%    CPU {self.__cpu_us:2.0f}%\n')
            sys.stdout.flush()
        return False