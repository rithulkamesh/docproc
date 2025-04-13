import csv
from typing import Any, Iterator, Optional, Callable, Dict, List, Set
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
    """

    def __init__(
        self, filename: str, progress_callback: Optional[Callable[[int], None]] = None
    ) -> None:
        """Initialize the CSV writer."""
        super().__init__(filename, progress_callback)
        self.file = None
        self.writer = None
        self._headers = None

        # Only include the fields specifically requested
        self._all_fields = {
            "record_type",
            "region_type",
            "content",
        }

        # Define region metadata fields separately
        self._region_metadata_fields = {
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "confidence",
            "page_number",  # Added page_number to include it in region metadata
        }

    def init_tables(self) -> None:
        """Initialize the output CSV file."""
        self.file = open(self.filename, "w", newline="")

    def write_data(self, data: Iterator[Dict[str, Any]]) -> None:
        """Write data rows to the CSV file."""
        batch_size = 1000
        batch_counter = 0
        try:
            first_row = next(data)
        except StopIteration:
            return

        if not self._headers:
            # Only include the specific fields required, plus region_metadata
            headers = ["record_type", "region_type", "content", "region_metadata"]

            self._headers = headers
            self.writer = csv.DictWriter(
                self.file, fieldnames=self._headers, extrasaction="ignore"
            )
            self.writer.writeheader()

        # Process the first row to handle region metadata appropriately
        processed_row = self._process_row(first_row)
        self.writer.writerow(processed_row)
        self._count += 1

        if self.progress_callback:
            self.progress_callback(self._count)

        # Track content for clubbing identical equations
        equation_content_map = {}
        if (
            "content" in first_row
            and "region_type" in first_row
            and first_row["region_type"] == "RegionType.EQUATION"
        ):
            equation_content_map[first_row["content"]] = True

        for row in data:
            processed_row = self._process_row(row)

            # Check if this is an equation that should be clubbed
            if (
                "content" in row
                and "region_type" in row
                and row["region_type"] == "RegionType.EQUATION"
                and row["content"] in equation_content_map
            ):
                # Skip writing duplicate equations
                continue
            elif (
                "content" in row
                and "region_type" in row
                and row["region_type"] == "RegionType.EQUATION"
            ):
                equation_content_map[row["content"]] = True

            self.writer.writerow(processed_row)
            self._count += 1
            batch_counter += 1

            if batch_counter >= batch_size:
                self.file.flush()  # flush to disk every batch
                batch_counter = 0
                if self.progress_callback:
                    self.progress_callback(self._count)

        self.file.flush()

    def _process_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Process a row to handle region metadata appropriately."""
        # Create a clean copy with only the required fields
        processed_row = {
            "record_type": row.get("record_type", ""),
            "region_type": row.get("region_type", ""),
            "content": row.get("content", ""),
        }

        # Handle region metadata fields
        if "record_type" in row and row["record_type"] in [
            "visual_content",
            "text_content",
        ]:
            region_metadata = {}
            for field in self._region_metadata_fields:
                if field in row:
                    region_metadata[field] = row[field]

            # Add region_metadata to the processed row if not empty
            if region_metadata:
                processed_row["region_metadata"] = str(region_metadata)

        # Escape newlines in text fields
        for key, value in processed_row.items():
            if isinstance(value, str):
                processed_row[key] = value.replace("\n", "\\n")

        return processed_row

    def close(self) -> None:
        """Close the CSV file."""
        if self.file:
            self.file.close()
