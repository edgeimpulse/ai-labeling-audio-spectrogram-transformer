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


from typing import Dict, List, Optional
from pydantic import BaseModel, Field, StrictBool, StrictInt, StrictStr
from edgeimpulse_api.models.get_synthetic_data_config_response_all_of_recent_jobs import GetSyntheticDataConfigResponseAllOfRecentJobs

class GetSyntheticDataConfigResponse(BaseModel):
    success: StrictBool = Field(..., description="Whether the operation succeeded")
    error: Optional[StrictStr] = Field(None, description="Optional error description (set if 'success' was false)")
    recent_jobs: List[GetSyntheticDataConfigResponseAllOfRecentJobs] = Field(..., alias="recentJobs")
    last_used_transformation_block_id: Optional[StrictInt] = Field(None, alias="lastUsedTransformationBlockId")
    last_used_parameters: Optional[Dict[str, StrictStr]] = Field(None, alias="lastUsedParameters")
    __properties = ["success", "error", "recentJobs", "lastUsedTransformationBlockId", "lastUsedParameters"]

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
    def from_json(cls, json_str: str) -> GetSyntheticDataConfigResponse:
        """Create an instance of GetSyntheticDataConfigResponse from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of each item in recent_jobs (list)
        _items = []
        if self.recent_jobs:
            for _item in self.recent_jobs:
                if _item:
                    _items.append(_item.to_dict())
            _dict['recentJobs'] = _items
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> GetSyntheticDataConfigResponse:
        """Create an instance of GetSyntheticDataConfigResponse from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return GetSyntheticDataConfigResponse.construct(**obj)

        _obj = GetSyntheticDataConfigResponse.construct(**{
            "success": obj.get("success"),
            "error": obj.get("error"),
            "recent_jobs": [GetSyntheticDataConfigResponseAllOfRecentJobs.from_dict(_item) for _item in obj.get("recentJobs")] if obj.get("recentJobs") is not None else None,
            "last_used_transformation_block_id": obj.get("lastUsedTransformationBlockId"),
            "last_used_parameters": obj.get("lastUsedParameters")
        })
        return _obj

