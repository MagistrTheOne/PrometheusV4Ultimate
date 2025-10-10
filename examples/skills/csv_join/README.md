# CSV Join Skill

## Description

The CSV Join skill merges two CSV files based on a common key column, supporting different join types (inner, left, right, outer).

## Features

- **Join Types**: inner, left, right, outer joins
- **Column Conflict Resolution**: Automatically renames conflicting columns
- **Error Handling**: Validates input files and key columns
- **Resource Limits**: CPU, memory, time, and disk usage limits

## Inputs

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `left_file` | string | Path to left CSV file | Yes |
| `right_file` | string | Path to right CSV file | Yes |
| `left_key` | string | Key column name in left CSV | Yes |
| `right_key` | string | Key column name in right CSV | Yes |
| `join_type` | string | Type of join: inner, left, right, outer (default: inner) | No |
| `output_file` | string | Path for output CSV file | Yes |

## Outputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `output_file` | string | Path to the created joined CSV file |
| `rows_count` | integer | Number of rows in the output |
| `columns_count` | integer | Number of columns in the output |

## Permissions

- **File System Read**: Yes
- **File System Write**: Yes
- **Network Access**: No
- **Environment Variables**: No

## Resource Limits

- **CPU**: 5000ms
- **Memory**: 200MB
- **Time**: 30s
- **Disk**: 100MB

## Examples

### Inner Join

```python
result = skill.run(
    left_file="users.csv",
    right_file="orders.csv",
    left_key="user_id",
    right_key="user_id",
    join_type="inner",
    output_file="user_orders.csv"
)
```

### Left Join

```python
result = skill.run(
    left_file="users.csv",
    right_file="orders.csv",
    left_key="user_id",
    right_key="user_id",
    join_type="left",
    output_file="all_users_with_orders.csv"
)
```

## Error Handling

The skill handles various error conditions:

- **Missing Files**: Returns error if input files don't exist
- **Missing Key Columns**: Returns error if specified key columns are not found
- **Empty Files**: Returns error if input CSV files are empty
- **Column Conflicts**: Automatically renames conflicting columns (e.g., `right_product`)

## Testing

Run tests with:

```bash
cd examples/skills/csv_join
python -m pytest tests/ -v
```

## License

MIT License - PrometheusULTIMATE
