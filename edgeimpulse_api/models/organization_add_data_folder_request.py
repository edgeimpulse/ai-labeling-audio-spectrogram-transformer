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
from pydantic import BaseModel, Field, StrictInt, StrictStr
from edgeimpulse_api.models.organization_dataset_type_enum import OrganizationDatasetTypeEnum

class OrganizationAddDataFolderRequest(BaseModel):
    dataset: StrictStr = ...
    bucket_id: StrictInt = Field(..., alias="bucketId")
    bucket_path: StrictStr = Field(..., alias="bucketPath")
    metadata_dataset: Optional[StrictStr] = Field(None, alias="metadataDataset")
    type: Optional[OrganizationDatasetTypeEnum] = None
    __properties = ["dataset", "bucketId", "bucketPath", "metadataDataset", "type"]

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
    def from_json(cls, json_str: str) -> OrganizationAddDataFolderRequest:
        """Create an instance of OrganizationAddDataFolderRequest from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> OrganizationAddDataFolderRequest:
        """Create an instance of OrganizationAddDataFolderRequest from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return OrganizationAddDataFolderRequest.construct(**obj)

        _obj = OrganizationAddDataFolderRequest.construct(**{
            "dataset": obj.get("dataset"),
            "bucket_id": obj.get("bucketId"),
            "bucket_path": obj.get("bucketPath"),
            "metadata_dataset": obj.get("metadataDataset"),
            "type": obj.get("type")
        })
        return _obj
