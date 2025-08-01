import mcp_playground.task_server as ts


def test_add_and_list_tasks(tmp_path, monkeypatch):
    tasks_file = tmp_path / 'tasks.txt'
    monkeypatch.setattr(ts, 'TASKS_FILE', str(tasks_file))

    ts.add_task('first')
    ts.add_task('second')
    tasks = ts.list_tasks()
    assert tasks == ['first', 'second']