import sqlite3
from typing import Any, Iterator, Optional, Callable, Dict, List
from .filewriter import FileWriter


class SQLiteWriter(FileWriter):
    """A writer class for exporting data to SQLite database format.

    Extends the base FileWriter class to provide SQLite-specific writing functionality.
    Handles table creation, schema inference, batch inserts, and progress tracking
    during the export process.

    Args:
        filename (str): Path to the output SQLite database file
        table_name (str): Name of the table to write data to
        progress_callback (Optional[Callable[[int], None]]): Optional callback function
            that accepts the current count of processed rows
        batch_size (int): Number of rows to insert in a single transaction (default: 1000)

    Attributes:
        conn: SQLite database connection
        cursor: SQLite cursor for executing queries
        _headers (List[str]): Column names detected from the first data row
        _count (int): Number of rows processed (inherited from FileWriter)
        _batch_size (int): Size of batch inserts
        _table_name (str): Name of the target table
        _column_types (Dict[str, str]): Mapping of column names to SQLite types

    Example:
        ```python
        def progress(count: int):
            print(f"Processed {count} rows")

        with SQLiteWriter("output.db", "my_table", progress) as writer:
            writer.init_tables()
            writer.write_data(data_iterator)
        ```
    """

    def __init__(
        self,
        filename: str,
        table_name: str,
        progress_callback: Optional[Callable[[int], None]] = None,
        batch_size: int = 1000,
    ) -> None:
        """Initialize the SQLite writer.

        Args:
            filename (str): Path to the output SQLite database file
            table_name (str): Name of the table to write data to
            progress_callback (Optional[Callable[[int], None]]): Optional callback function
                to report progress
            batch_size (int): Number of rows to insert in a single transaction
        """
        super().__init__(filename, progress_callback)
        self.conn = None
        self.cursor = None
        self._headers = None
        self._batch_size = batch_size
        self._table_name = table_name
        self._column_types = {}
        self._current_batch = []

    def _infer_sqlite_type(self, value: Any) -> str:
        """Infer SQLite column type from Python value."""
        if isinstance(value, (int, bool)):
            return "INTEGER"
        elif isinstance(value, float):
            return "REAL"
        elif isinstance(value, (str, dict, list)) or value is None:
            return "TEXT"  # Store all complex types as JSON text
        else:
            return "BLOB"

    def _create_table(self, columns: List[str], types: Dict[str, str]) -> None:
        """Create the SQLite table with the given schema.

        Args:
            columns (List[str]): List of column names
            types (Dict[str, str]): Mapping of column names to SQLite types

        Raises:
            sqlite3.Error: If table creation fails
        """
        columns_def = ", ".join(f'"{col}" {types[col]}' for col in columns)
        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS "{self._table_name}" (
                {columns_def}
            )
        """
        self.cursor.execute(create_table_sql)
        self.conn.commit()

    def init_tables(self) -> None:
        """Initialize the SQLite database connection.

        Opens the database connection and creates a cursor.
        Should be called before write_data().

        Raises:
            sqlite3.Error: If database connection fails
        """
        self.conn = sqlite3.connect(self.filename)
        self.cursor = self.conn.cursor()

    def _flush_batch(self) -> None:
        """Write the current batch of rows to the database.

        Executes a batch insert of accumulated rows and clears the batch.
        """
        if not self._current_batch:
            return

        placeholders = ",".join(["?" for _ in self._headers])
        insert_sql = f"""
            INSERT INTO "{self._table_name}" 
            ({",".join(f'"{h}"' for h in self._headers)})
            VALUES ({placeholders})
        """

        self.cursor.executemany(
            insert_sql,
            [tuple(row[h] for h in self._headers) for row in self._current_batch],
        )
        self.conn.commit()
        self._current_batch = []

    def write_data(self, data: Iterator[Dict[str, Any]]) -> None:
        """Write data rows to the SQLite database.

        Processes an iterator of dictionary rows, automatically detecting schema
        from the first row if not already set. Performs batch inserts for efficiency
        and updates progress through the callback if provided.

        Args:
            data (Iterator[Dict[str, Any]]): Iterator yielding dictionaries where
                keys are column names and values are cell values

        Note:
            - Table schema is determined from the first row
            - Empty iterators are handled gracefully
            - Progress callback is called after each batch is written
            - Rows are inserted in batches for better performance

        Raises:
            sqlite3.Error: If there are database operation issues
            ValueError: If row data doesn't match schema
        """
        try:
            first_row = next(data)
            if not self._headers:
                self._headers = list(first_row.keys())
                self._column_types = {
                    col: self._infer_sqlite_type(first_row[col])
                    for col in self._headers
                }
                self._create_table(self._headers, self._column_types)

            self._current_batch.append(first_row)
            self._count += 1

            for row in data:
                self._current_batch.append(row)
                self._count += 1

                if len(self._current_batch) >= self._batch_size:
                    self._flush_batch()
                    if self.progress_callback:
                        self.progress_callback(self._count)

            # Flush any remaining rows
            if self._current_batch:
                self._flush_batch()
                if self.progress_callback:
                    self.progress_callback(self._count)

        except StopIteration:
            if not self._headers:
                self._headers = []
                self._column_types = {}
                self._create_table(self._headers, self._column_types)

    def close(self) -> None:
        """Close the SQLite database connection.

        Ensures proper cleanup by:
        1. Flushing any remaining batched rows
        2. Closing the cursor
        3. Closing the database connection
        Safe to call multiple times.
        """
        if self._current_batch:
            self._flush_batch()

        if self.cursor:
            self.cursor.close()

        if self.conn:
            self.conn.close()
