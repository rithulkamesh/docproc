import csv
from typing import Any, Iterator, Optional, Callable, Dict
from .filewriter import FileWriter


class CSVWriter(FileWriter):
    """A writer class for exporting data to CSV format.

    Extends the base FileWriter class to provide CSV-specific writing functionality.
    Handles dynamic header detection, row writing, and progress tracking during
    the export process.

    Args:
        filename (str): Path to the output CSV file
        progress_callback (Optional[Callable[[int], None]]): Optional callback function
            that accepts the current count of processed rows

    Attributes:
        file: File handle for the output CSV file
        writer: CSV DictWriter instance for writing rows
        _headers (List[str]): Column headers detected from the first data row
        _count (int): Number of rows processed (inherited from FileWriter)

    Example:
        ```python
        def progress(count: int):
            print(f"Processed {count} rows")

        with CSVWriter("output.csv", progress) as writer:
            writer.init_tables()
            writer.write_data(data_iterator)
        ```
    """

    def __init__(
        self, filename: str, progress_callback: Optional[Callable[[int], None]] = None
    ) -> None:
        """Initialize the CSV writer.

        Args:
            filename (str): Path to the output CSV file
            progress_callback (Optional[Callable[[int], None]]): Optional callback function
                to report progress
        """
        super().__init__(filename, progress_callback)
        self.file = None
        self.writer = None
        self._headers = None

    def init_tables(self) -> None:
        """Initialize the output CSV file.

        Opens the output file in write mode with proper newline handling.
        Should be called before write_data().

        Raises:
            IOError: If the file cannot be opened for writing
        """
        self.file = open(self.filename, "w", newline="")

    def write_data(self, data: Iterator[Dict[str, Any]]) -> None:
        """Write data rows to the CSV file.

        Processes an iterator of dictionary rows, automatically detecting headers
        from the first row if not already set. Updates progress through the
        callback if provided.

        Args:
            data (Iterator[Dict[str, Any]]): Iterator yielding dictionaries where
                keys are column names and values are cell values

        Note:
            - Headers are determined from the keys of the first row
            - Empty iterators are handled gracefully
            - Progress callback is called after each row is written

        Raises:
            IOError: If there are issues writing to the file
            ValueError: If row data doesn't match headers
        """
        batch_size = 1000
        batch_counter = 0
        try:
            first_row = next(data)
        except StopIteration:
            return

        if not self._headers:
            self._headers = list(first_row.keys())
            self.writer = csv.DictWriter(self.file, fieldnames=self._headers)
            self.writer.writeheader()

        # Escape newlines in the first row.
        safe_row = {
            k: v.replace("\n", "\\n") if isinstance(v, str) else v
            for k, v in first_row.items()
        }
        self.writer.writerow(safe_row)
        self._count += 1
        batch_counter += 1
        if self.progress_callback:
            self.progress_callback(self._count)

        for row in data:
            safe_row = {
                k: v.replace("\n", "\\n") if isinstance(v, str) else v
                for k, v in row.items()
            }
            self.writer.writerow(safe_row)
            self._count += 1
            batch_counter += 1
            if batch_counter >= batch_size:
                self.file.flush()  # flush to disk every batch
                batch_counter = 0
                if self.progress_callback:
                    self.progress_callback(self._count)
        self.file.flush()

    def close(self) -> None:
        """Close the CSV file.

        Ensures proper cleanup by closing the output file if it's open.
        Safe to call multiple times.
        """
        if self.file:
            self.file.close()
