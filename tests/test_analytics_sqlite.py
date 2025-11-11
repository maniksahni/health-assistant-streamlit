import os
import tempfile
from data.analytics import get_conn, record_visit, query_summary


def test_sqlite_record_and_summary(tmp_path):
    base = tmp_path
    conn, kind = get_conn(str(base))
    assert kind == 'sqlite'
    vid = 'visitor-1'
    record_visit(conn, kind, vid, 'Chat with HealthBot', 'ua', '127.0.0.1')
    record_visit(conn, kind, vid, 'Chat with HealthBot', 'ua', '127.0.0.1')
    s = query_summary(conn, kind)
    assert s['unique_visitors'] >= 1
    assert s['total_visits'] >= 2
