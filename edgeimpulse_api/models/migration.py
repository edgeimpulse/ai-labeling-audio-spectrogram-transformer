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


from typing import Optional
from pydantic import BaseModel, Field, StrictInt, StrictStr, validator

class Migration(BaseModel):
    id: StrictStr = Field(..., description="Unique identifier of the data migration")
    state: StrictStr = Field(..., description="Migration state. Can be 'paused', 'queued', 'running', 'done', 'failed'")
    offset: Optional[StrictInt] = Field(None, description="Number of items already processed")
    __properties = ["id", "state", "offset"]

    @validator('state')
    def state_validate_enum(cls, v):
        if v not in ('paused', 'queued', 'running', 'done', 'failed'):
            raise ValueError("must validate the enum values ('paused', 'queued', 'running', 'done', 'failed')")
        return v

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
    def from_json(cls, json_str: str) -> Migration:
        """Create an instance of Migration from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> Migration:
        """Create an instance of Migration from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return Migration.construct(**obj)

        _obj = Migration.construct(**{
            "id": obj.get("id"),
            "state": obj.get("state"),
            "offset": obj.get("offset")
        })
        return _obj

