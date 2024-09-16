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


from typing import List
from pydantic import BaseModel, StrictStr
from edgeimpulse_api.models.organization_add_dataset_request_bucket import OrganizationAddDatasetRequestBucket
from edgeimpulse_api.models.organization_dataset_type_enum import OrganizationDatasetTypeEnum

class OrganizationAddDatasetRequest(BaseModel):
    dataset: StrictStr = ...
    tags: List[StrictStr] = ...
    category: StrictStr = ...
    type: OrganizationDatasetTypeEnum = ...
    bucket: OrganizationAddDatasetRequestBucket = ...
    __properties = ["dataset", "tags", "category", "type", "bucket"]

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
    def from_json(cls, json_str: str) -> OrganizationAddDatasetRequest:
        """Create an instance of OrganizationAddDatasetRequest from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of bucket
        if self.bucket:
            _dict['bucket'] = self.bucket.to_dict()
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> OrganizationAddDatasetRequest:
        """Create an instance of OrganizationAddDatasetRequest from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return OrganizationAddDatasetRequest.construct(**obj)

        _obj = OrganizationAddDatasetRequest.construct(**{
            "dataset": obj.get("dataset"),
            "tags": obj.get("tags"),
            "category": obj.get("category"),
            "type": obj.get("type"),
            "bucket": OrganizationAddDatasetRequestBucket.from_dict(obj.get("bucket")) if obj.get("bucket") is not None else None
        })
        return _obj
