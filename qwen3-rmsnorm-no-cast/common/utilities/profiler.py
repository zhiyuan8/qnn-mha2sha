"""common utilities and class implementation for time/GPU/RAM profiling"""


## NEXA: Essentially, we can use the following code to use the profiler.
#   with event_marker("loading_data"):
#       do something

import contextlib
import gc
import os
import time
from datetime import timedelta
from typing import Union, Optional
import json
from multiprocessing import Process, Value, RLock

from dataclasses import dataclass
import psutil
from enum import Enum

import logging
logger = logger = logging.getLogger(__name__)
logging.basicConfig(filename='./logfile.log', level=logging.INFO)

class Device(Enum):
    CPU = 1
    GPU = 2
    NSP = 3

WATERMARK_THREAD_POLLING_INTERVAL_IN_MS = 100
def convert_bytes(size: int):
    """
    :return bytes in human-readable format
    """
    sign = ''
    if size < 0:
        sign = '-'
        size = abs(size)

    for unit in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            break
        size /= 1024.0
    return "%s%3.1f %s" % (sign, size, unit)

@dataclass
class ProfileMarker:
    """ Implements methods to capture profiling data """
    event: str
    device: Optional[int]
    cpu_memory_usage: int
    delta_cpu_memory_usage: int
    time: float

    def __str__(self):
        """ string representation of the marker event """
        return f'Event {self.event} : time={timedelta(seconds=self.time)}, ' \
               f'RAM={convert_bytes(self.delta_cpu_memory_usage)}(+ {convert_bytes(self.cpu_memory_usage)})'

    def to_dict(self) -> dict:
        """return data in dict """
        return {
            'event': self.event,
            'device': self.device,
            'cpu_memory_usage': self.cpu_memory_usage,
            'delta_cpu_memory_usage': self.delta_cpu_memory_usage,
            'time': self.time
        }

def ram_usage(pid:int, include_children:bool = True):
    """ :return RAM usage for given process Id and its childeren. """
    """
      VSS (reported as VSZ from ps) is the total accessible address space of a process.
      This size also includes memory that may not be resident in RAM like mallocs that have been allocated but not written to.
      VSS is of very little use for determing real memory usage of a process.
     
      RSS is the total memory actually held in RAM for a process. RSS can be misleading,
      because it reports the total all of the shared libraries that the process uses,
      even though a shared library is only loaded into memory once regardless of how many processes use it.
      RSS is not an accurate representation of the memory usage for a single process.
     
      PSS differs from RSS in that it reports the proportional size of its shared libraries,
      i.e. if three processes all use a shared library that has 30 pages, that library will only contribute
      10 pages to the PSS that is reported for each of the three processes. PSS is a very useful number
      because when the PSS for all processes in the system are summed together,
      that is a good representation for the total memory usage in the system. When a process is killed, the shared libraries that
      contributed to its PSS will be proportionally distributed to the PSS totals for the remaining processes still using that library.
      In this way PSS can be slightly misleading, because when a process is killed, PSS does not accurately represent the memory returned to the overall system.
 
      USS is the total private memory for a process, i.e. that memory that is completely unique to that process.
        USS is an extremely useful number because it indicates the true incremental cost of running a particular process. When a process is killed,
        the USS is the total memory that is actually returned to the system. USS is the best number to watch when initially suspicious of memory leaks in a process.
    """
   
    parent = psutil.Process(pid)
    polling_pid = os.getpid()
    try:
        ram = psutil.Process(pid).memory_full_info().pss
    except:
        ram = psutil.Process(pid).memory_info().rss
    total_ram = [ram]
 
    children = parent.children(recursive=True)
 
    for child in children:
        # exclude the polling process if the polling process was created and used now
        #or, if this function is called not in polling process context, all the children should be counted
        if WATERMARK_THREAD_POLLING_INTERVAL_IN_MS == 0 or child.pid != polling_pid:
            try:
                ram = psutil.Process(child.pid).memory_full_info().pss
            except:
                ram=0
            total_ram.append(ram)
    return sum(total_ram)

@dataclass
class EventMarker:
    """ Implements methods to capture system stats during an event """
    event: str
    device: Optional[int]
    _cpu_memory_usage: int
    time = time.time()

    def __init__(self, event: str = None, cpu_memory_usage: int = 0, device=None):
        self._cpu_memory_usage: int = cpu_memory_usage
        self.device = device
        self.time = time.time()
        if event is None:
            self.event = f'@ {self.time}'
        else:
            self.event = event

    def delta(self, event: str, start_marker: 'EventMarker') -> ProfileMarker:
        """computes diff between two event marker and returns profile marker"""
        # pylint: disable=protected-access
        return ProfileMarker(
            event,
            self.device,
            start_marker._cpu_memory_usage,
            self._cpu_memory_usage - start_marker._cpu_memory_usage,
            int(self.time - start_marker.time)
        )

    def __str__(self):
        """ string representation of the marker event """
        return f'Event {self.event} : time={self.time}, RAM={self.cpu_memory_usage}'


    @property
    def cpu_memory_usage(self) -> str:
        """returns a string representation of RAM usage """
        return convert_bytes(self._cpu_memory_usage)

    def to_dict(self) -> dict:
        """return data in dict """
        return {
            'event': self.event,
            'device': self.device,
            'cpu_memory_usage': self._cpu_memory_usage,
            'time': self.time
        }

def ram_watermark_function(ram_allocated:Value, pid: int, polling_interval_in_ms:float):
    """
    observing process to reflect current peak RAM usage.
    :param ram_allocated: shared variable used by profiler to reset to new allocation and the observing(this) process
    to track max allocation.
    :param pid: parent process pid for tracking mem allocation
    :param polling_interval_in_ms: interval between polling for memory usage
    """
    logger.info('Created RAM watermark daemon process(pid=%d) for pid=%d, polling at %.1f ms',
                os.getpid(), pid, polling_interval_in_ms)
    while psutil.pid_exists(pid):
        new_usage = ram_usage(pid)
        with ram_allocated.get_lock():
            ram_allocated.value = max(new_usage, ram_allocated.value)
        time.sleep(polling_interval_in_ms/ 1000.0)


# pylint: disable=no-member
class EventProfiler:
    """ Implements methods to profile latency and RAM memory usage """
    _instance = None

    def __new__(cls):
        """ Implements the Global Object Pattern  (Singleton) """
        if cls._instance is None:
            cls._instance = super(EventProfiler, cls).__new__(cls)
            cls._instance._empty_cache = False # pylint: disable=protected-access
            cls._instance._markers = [] # pylint: disable=protected-access

            if WATERMARK_THREAD_POLLING_INTERVAL_IN_MS:
                cls._instance._ram_allocated = Value('q', 0, lock=RLock()) # pylint: disable=protected-access
                p = Process(
                    target=ram_watermark_function,
                    args=(cls._instance._ram_allocated,
                          os.getpid(),
                          WATERMARK_THREAD_POLLING_INTERVAL_IN_MS
                          ))
                p.daemon = True # < will terminate the watermark process when this process exits
                cls._instance.reset_peak_memory_stats()
                p.start()

            logger.info('Created Latency/Memory profiler: empty_cache=%s',
                        cls._instance._empty_cache) # pylint: disable=protected-access

        return cls._instance

    def reset_peak_memory_stats(self):
        """ reset RAM usage to current RAM allocation. """
        if WATERMARK_THREAD_POLLING_INTERVAL_IN_MS:
            with self._ram_allocated.get_lock():
                self._ram_allocated.value = ram_usage(os.getpid())

    @property
    def max_memory_allocated(self):
        """ getter for current peak RAM usage since last reset. """
        if WATERMARK_THREAD_POLLING_INTERVAL_IN_MS:
            with self._ram_allocated.get_lock():
                return self._ram_allocated.value
        else:
            return ram_usage(os.getpid())


    @property
    def empty_cache(self):
        """ getter for empty_cache if set to True snapshot calls would flush unused CUDA memory. """
        return self._empty_cache

    @empty_cache.setter
    def empty_cache(self, enable: bool = False):
        """ setter for empty_cache if set to True snapshot calls would flush unused CUDA memory. """
        if self._empty_cache != enable:
            if self._empty_cache:
                logger.warning('enabling cache clear might impact latency, avoid excessive calls in tight loop')
            self._markers.append(f"empty_cache:{self._empty_cache}")


    def snapshot(self, snapshot_marker: str = None,
                 device: Union[Device, int] = None,
                 append: bool = True) -> EventMarker:
        """
        logs the current time and memory usage across all CUDA devices
        :param snapshot_marker: text to capture with the GPU marker.
        :param device: (torch.device or int, optional): selected device.
        :param append: if True, added it to report logs
        """
        marker = EventMarker(snapshot_marker, self.max_memory_allocated, device)

        logger.info("memory usage @ '%s' :  RAM %s",
                    snapshot_marker,  marker.cpu_memory_usage)
        if append:
            self._markers.append(marker)

        return marker

    def report(self):
        """ dumps the collected memory usage logs """
        for i, marker in enumerate(self._markers):
            logger.info("#%d: %s", i, marker)

        # For each marker, we have readings of both entry and exit (entry + delta) memory
        # Peak device memory is the max of these (2 * len(self._markers)) memory readings
        def get_device_peak_memory_stats(entry_mem_fn, delta_mem_fn):
            profile_markers = [m for m in self._markers if isinstance(m, ProfileMarker)]
            event_entry_exit_memory = [[entry_mem_fn(m), entry_mem_fn(m)+delta_mem_fn(m)] for m in profile_markers]
            max_mem = max(sum(event_entry_exit_memory, []))
            max_memory_event = []
            for m, (entry_mem, exit_mem) in zip(profile_markers, event_entry_exit_memory):
                if max_mem in (entry_mem, exit_mem):
                    max_memory_event.append(m.event)
            return max_mem, max_memory_event

        max_mem_cpu, max_memory_event_cpu = get_device_peak_memory_stats(lambda m : m.cpu_memory_usage, lambda m : m.delta_cpu_memory_usage)

        if len(self._markers) > 0:
            logger.info("Max CPU memory used: %s bytes for event(s) %s", convert_bytes(max_mem_cpu), (', '.join(max_memory_event_cpu)))


    def json_dump(self, filepath: str):
        """ dumps the collected memory usage into a json file """
        #system config
        mem_info = psutil.virtual_memory()
        swap_info = psutil.swap_memory()
        system_capacity = {
            "CPU cores": psutil.cpu_count(),
            "CPU freq(MHz)": psutil.cpu_freq().max,
            "total RAM (GB)": mem_info.total/(1024**3),
            "swap size (GB)": swap_info.total/(1024**3)
        }
        markers = [m.to_dict() for m in self._markers]
        result={
            "system_capacity":system_capacity,
            "event_markers":markers
        }
        with open(filepath, 'w') as f:
            json.dump(result,f,sort_keys=True, indent=4)

    def __exit__(self, exc_type, exc_value, traceback):
        pass

@contextlib.contextmanager
def event_marker(event: str, device: Union[Device, int] = None, flush_ram: bool = False):
    """
    utility to mark time taken and memory usage before and after executing a section of code.
    :param event: marker string to use to identify the context.
    :param device: (torch.device or int, optional): selected device.
    :param flush_ram: invoke garbage collect for true estimates before profiling.
    """
    profiler = EventProfiler()
    # reset for start low-watermark
    if flush_ram:
        gc.collect()
        event  = f'{event}[gc]'
    profiler.reset_peak_memory_stats()
    start_marker = profiler.snapshot(f'{event} >> ', device, append=False)
    yield
    end_marker = profiler.snapshot(f'{event} << ', device, append=False)
    profile_marker = end_marker.delta(event, start_marker)
    logger.info('%s', profile_marker)
    profiler._markers.append(profile_marker) # pylint: disable=protected-access
