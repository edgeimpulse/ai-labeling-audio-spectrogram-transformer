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



from pydantic import BaseModel, Field
from edgeimpulse_api.models.organization_transformation_block import OrganizationTransformationBlock

class GetOrganizationTransformationBlockResponseAllOf(BaseModel):
    transformation_block: OrganizationTransformationBlock = Field(..., alias="transformationBlock")
    __properties = ["transformationBlock"]

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
    def from_json(cls, json_str: str) -> GetOrganizationTransformationBlockResponseAllOf:
        """Create an instance of GetOrganizationTransformationBlockResponseAllOf from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of transformation_block
        if self.transformation_block:
            _dict['transformationBlock'] = self.transformation_block.to_dict()
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> GetOrganizationTransformationBlockResponseAllOf:
        """Create an instance of GetOrganizationTransformationBlockResponseAllOf from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return GetOrganizationTransformationBlockResponseAllOf.construct(**obj)

        _obj = GetOrganizationTransformationBlockResponseAllOf.construct(**{
            "transformation_block": OrganizationTransformationBlock.from_dict(obj.get("transformationBlock")) if obj.get("transformationBlock") is not None else None
        })
        return _obj

