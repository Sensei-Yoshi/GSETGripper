"""Material-only visual descriptor contract."""

from __future__ import annotations

from modules.perception import Description


def test_description_schema_excludes_contact_geometry() -> None:
    properties = Description.model_json_schema()["properties"]

    assert "contact_region" not in properties
    assert "local_geometry" not in properties
    assert "contact_material" in properties
    assert "visible_surface_condition" in properties


def test_legacy_descriptor_geometry_is_ignored() -> None:
    description = Description.model_validate(
        {
            "retrieval_description": "Clean, dry paperboard contact surface.",
            "contact_material": "paperboard",
            "contact_region": "legacy lateral band",
            "local_geometry": "legacy curved surface",
        }
    )

    assert description.description == "Clean, dry paperboard contact surface."
    assert description.contact_material == "paperboard"
    assert not hasattr(description, "contact_region")
    assert not hasattr(description, "local_geometry")
