import json
from typing import Any, Iterator, Optional, Callable, Dict
from .filewriter import FileWriter


class JSONWriter(FileWriter):
    """
    A writer class for exporting data to JSON Lines format.

    Extends the FileWriter base class to provide JSON-specific functionality
    for writing data incrementally. Each data record is written as a JSON object
    on a separate line, which is memory efficient for large datasets.

    Args:
        filename (str): Path to the output JSON file
        progress_callback (Optional[Callable[[int], None]]): Optional callback function
            that reports the number of processed records

    Attributes:
        file: File handle for the output JSON file
        _count (int): Number of records processed

    Example:
        def progress(count: int):
            print(f"Processed {count} records")

        with JSONWriter("output.json", progress) as writer:
            writer.init_tables()
            writer.write_data(data_iterator)
    """

    def __init__(
        self, filename: str, progress_callback: Optional[Callable[[int], None]] = None
    ) -> None:
        """
        Initialize the JSON writer.

        Sets up the filename and optional progress callback.

        Args:
            filename (str): Path to the output JSON file
            progress_callback (Optional[Callable[[int], None]]): Optional callback function
                to report progress
        """
        super().__init__(filename, progress_callback)
        self.file = None

    def init_tables(self) -> None:
        """
        Initialize the JSON writer.

        Opens the output file in write mode.
        Should be called before write_data().

        Raises:
            IOError: If the file cannot be opened for writing
        """
        self.file = open(self.filename, "w")

    def write_data(self, data: Iterator[Dict[str, Any]]) -> None:
        """
        Write data records to the JSON file in JSON Lines format.

        Processes an iterator of dictionaries, writing each record as a JSON object
        on a separate line. Updates the progress via the provided callback.

        Args:
            data (Iterator[Dict[str, Any]]): Iterator yielding dictionaries
                representing data records

        Raises:
            IOError: If there are issues writing to the file
        """
        batch_size = 1000
        batch_counter = 0
        for row in data:
            self.file.write(json.dumps(row) + "\n")
            self._count += 1
            batch_counter += 1
            if batch_counter >= batch_size:
                self.file.flush()  # flush buffer incrementally
                batch_counter = 0
                if self.progress_callback:
                    self.progress_callback(self._count)
        self.file.flush()

    def close(self) -> None:
        """
        Close the JSON file.

        Ensures the output file is properly closed.
        Safe to call multiple times.
        """
        if self.file:
            self.file.close()
