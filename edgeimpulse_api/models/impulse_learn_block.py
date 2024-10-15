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

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, StrictInt, StrictStr, conint
from edgeimpulse_api.models.learn_block_type import LearnBlockType

class ImpulseLearnBlock(BaseModel):
    id: conint(strict=True, ge=1) = Field(..., description="Identifier for this block. Make sure to up this number when creating a new block, and don't re-use identifiers. If the block hasn't changed, keep the ID as-is. ID must be unique across the project and greather than zero (>0).")
    type: LearnBlockType = ...
    name: StrictStr = Field(..., description="Block name, will be used in menus. If a block has a baseBlockId, this field is ignored and the base block's name is used instead.")
    dsp: List[StrictInt] = Field(..., description="DSP dependencies, identified by DSP block ID")
    title: StrictStr = Field(..., description="Block title, used in the impulse UI")
    description: Optional[StrictStr] = Field(None, description="A short description of the block version, displayed in the block versioning UI")
    created_by: Optional[StrictStr] = Field(None, alias="createdBy", description="The system component that created the block version (createImpulse | clone | tuner). Cannot be set via API.")
    created_at: Optional[datetime] = Field(None, alias="createdAt", description="The datetime that the block version was created. Cannot be set via API.")
    __properties = ["id", "type", "name", "dsp", "title", "description", "createdBy", "createdAt"]

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
    def from_json(cls, json_str: str) -> ImpulseLearnBlock:
        """Create an instance of ImpulseLearnBlock from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> ImpulseLearnBlock:
        """Create an instance of ImpulseLearnBlock from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return ImpulseLearnBlock.construct(**obj)

        _obj = ImpulseLearnBlock.construct(**{
            "id": obj.get("id"),
            "type": obj.get("type"),
            "name": obj.get("name"),
            "dsp": obj.get("dsp"),
            "title": obj.get("title"),
            "description": obj.get("description"),
            "created_by": obj.get("createdBy"),
            "created_at": obj.get("createdAt")
        })
        return _obj
