"""Material-only visual descriptor contract."""

from __future__ import annotations

from modules.perception import Description


def test_description_schema_excludes_contact_geometry() -> None:
    properties = Description.model_json_schema()["properties"]

    assert "contact_region" not in properties
    assert "local_geometry" not in properties
    assert "contact_material" in properties
    assert "visible_surface_condition" in properties

