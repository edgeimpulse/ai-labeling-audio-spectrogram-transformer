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
from pydantic import BaseModel, Field, StrictBool, StrictStr
from edgeimpulse_api.models.organization_dsp_block import OrganizationDspBlock

class GetOrganizationDspBlockResponse(BaseModel):
    success: StrictBool = Field(..., description="Whether the operation succeeded")
    error: Optional[StrictStr] = Field(None, description="Optional error description (set if 'success' was false)")
    dsp_block: OrganizationDspBlock = Field(..., alias="dspBlock")
    __properties = ["success", "error", "dspBlock"]

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
    def from_json(cls, json_str: str) -> GetOrganizationDspBlockResponse:
        """Create an instance of GetOrganizationDspBlockResponse from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of dsp_block
        if self.dsp_block:
            _dict['dspBlock'] = self.dsp_block.to_dict()
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> GetOrganizationDspBlockResponse:
        """Create an instance of GetOrganizationDspBlockResponse from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return GetOrganizationDspBlockResponse.construct(**obj)

        _obj = GetOrganizationDspBlockResponse.construct(**{
            "success": obj.get("success"),
            "error": obj.get("error"),
            "dsp_block": OrganizationDspBlock.from_dict(obj.get("dspBlock")) if obj.get("dspBlock") is not None else None
        })
        return _obj

