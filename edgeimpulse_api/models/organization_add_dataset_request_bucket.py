# coding: utf-8

"""
    Edge Impulse API

    No description provided (generated by Openapi Generator https://github.com/openapitools/openapi-generator)  # noqa: E501

    The version of the OpenAPI document: 1.0.0
    Generated by: https://openapi-generator.tech
"""


from __future__ import annotations
from inspect import getfullargspec
import pprint
import re  # noqa: F401
import json



from pydantic import BaseModel, Field, StrictInt, StrictStr

class OrganizationAddDatasetRequestBucket(BaseModel):
    id: StrictInt = Field(..., description="Bucket ID")
    path: StrictStr = Field(..., description="Path in the bucket")
    data_item_naming_levels_deep: StrictInt = Field(..., alias="dataItemNamingLevelsDeep", description="Number of levels deep for data items, e.g. if you have folder \"test/abc\", with value 1 \"test\" will be a data item, with value 2 \"test/abc\" will be a data item. Only used for \"clinical\" type.")
    __properties = ["id", "path", "dataItemNamingLevelsDeep"]

    class Config:
        allow_population_by_field_name = True
        validate_assignment = False

    def to_str(self) -> str:
        """Returns the string representation of the model using alias"""
        return pprint.pformat(self.dict(by_alias=True))

    def to_json(self) -> str:
        """Returns the JSON representation of the model using alias"""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> OrganizationAddDatasetRequestBucket:
        """Create an instance of OrganizationAddDatasetRequestBucket from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> OrganizationAddDatasetRequestBucket:
        """Create an instance of OrganizationAddDatasetRequestBucket from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return OrganizationAddDatasetRequestBucket.construct(**obj)

        _obj = OrganizationAddDatasetRequestBucket.construct(**{
            "id": obj.get("id"),
            "path": obj.get("path"),
            "data_item_naming_levels_deep": obj.get("dataItemNamingLevelsDeep")
        })
        return _obj
