from abc import ABC, abstractmethod

from typing import Any, Iterator, Optional, Callable, Dict


class FileWriter(ABC):
    """Abstract base class for file writers that support incremental data writing.

    Defines the interface for writers that can export data to various file formats.
    Provides context manager support and progress tracking functionality.

    This class should be subclassed to implement specific file format writers
    (e.g., CSV, JSON, XML). All abstract methods must be implemented by subclasses.

    Args:
        filename (str): Path to the output file
        progress_callback (Optional[Callable[[int], None]]): Optional callback function
            that accepts the current count of processed items

    Attributes:
        filename (str): Path to the output file
        progress_callback (Optional[Callable[[int], None]]): Callback for progress updates
        _count (int): Number of items processed

    Example:
        ```python
        class CSVWriter(FileWriter):
            def __init__(self, filename: str,
                        progress_callback: Optional[Callable[[int], None]] = None) -> None:
                super().__init__(filename, progress_callback)
                # CSV-specific initialization

            def write_data(self, data: Iterator[Dict[str, Any]]) -> None:
                # Implement CSV writing logic
                pass

            def init_tables(self) -> None:
                # Initialize CSV file
                pass

            def close(self) -> None:
                # Close CSV file
                pass
        ```
    """

    @abstractmethod
    def __init__(
        self, filename: str, progress_callback: Optional[Callable[[int], None]] = None
    ) -> None:
        """Initialize the file writer.

        Sets up basic writer attributes including filename and progress tracking.
        Subclasses must call this method via super().__init__().

        Args:
            filename (str): Path to the output file
            progress_callback (Optional[Callable[[int], None]]): Optional callback
                function to report progress
        """
        self.filename = filename
        self.progress_callback = progress_callback
        self._count = 0

    @abstractmethod
    def write_data(self, data: Iterator[Dict[str, Any]]) -> None:
        """Write data incrementally from an iterator.

        Processes an iterator of dictionary items, writing them to the output file
        in the appropriate format. Should handle progress tracking via the callback
        if provided.

        Args:
            data (Iterator[Dict[str, Any]]): Iterator yielding dictionaries where
                keys represent field names and values represent field values

        Note:
            - Implementation should be memory efficient for large datasets
            - Should update self._count and call progress_callback as appropriate
            - Should handle empty iterators gracefully
        """
        pass

    @abstractmethod
    def init_tables(self) -> None:
        """Initialize any required storage structures.

        Performs any necessary setup before writing data, such as:
        - Creating output files
        - Writing headers
        - Initializing database tables
        - Setting up network connections

        Should be called before write_data().

        Note:
            Implementation should handle cases where initialization has already
            occurred gracefully.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up resources.

        Performs necessary cleanup operations such as:
        - Closing file handles
        - Committing transactions
        - Closing network connections

        Note:
            - Implementation should be idempotent (safe to call multiple times)
            - Should handle cases where resources were never initialized
        """
        pass

    def __enter__(self):
        """Context manager entry.

        Enables use of the writer in a 'with' statement.

        Returns:
            FileWriter: The writer instance
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit.

        Ensures proper cleanup by calling close() when exiting the 'with' block,
        even if an exception occurred.

        Args:
            exc_type: Type of exception that occurred, if any
            exc_val: Exception instance that occurred, if any
            exc_tb: Traceback information, if an exception occurred
        """
        self.close()
