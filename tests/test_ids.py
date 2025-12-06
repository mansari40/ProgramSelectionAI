from src.utils.ids import make_chunk_id, make_point_id


def test_ids_are_stable():
    ch = make_chunk_id("pdf", "CSSE_MSC_on_campus", 0)
    assert ch == "pdf::CSSE_MSC_on_campus::0"

    a = make_point_id(ch)
    b = make_point_id(ch)
    assert a == b
