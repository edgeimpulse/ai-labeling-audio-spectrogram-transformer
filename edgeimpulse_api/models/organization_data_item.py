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
from typing import Dict, List
from pydantic import BaseModel, Field, StrictInt, StrictStr
from edgeimpulse_api.models.organization_data_item_files_inner import OrganizationDataItemFilesInner

class OrganizationDataItem(BaseModel):
    id: StrictInt = ...
    name: StrictStr = ...
    bucket_id: StrictInt = Field(..., alias="bucketId")
    bucket_name: StrictStr = Field(..., alias="bucketName")
    bucket_path: StrictStr = Field(..., alias="bucketPath")
    dataset: StrictStr = ...
    total_file_count: StrictInt = Field(..., alias="totalFileCount")
    total_file_size: StrictInt = Field(..., alias="totalFileSize")
    created: datetime = ...
    metadata: Dict[str, StrictStr] = ...
    files: List[OrganizationDataItemFilesInner] = ...
    __properties = ["id", "name", "bucketId", "bucketName", "bucketPath", "dataset", "totalFileCount", "totalFileSize", "created", "metadata", "files"]

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
    def from_json(cls, json_str: str) -> OrganizationDataItem:
        """Create an instance of OrganizationDataItem from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of each item in files (list)
        _items = []
        if self.files:
            for _item in self.files:
                if _item:
                    _items.append(_item.to_dict())
            _dict['files'] = _items
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> OrganizationDataItem:
        """Create an instance of OrganizationDataItem from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return OrganizationDataItem.construct(**obj)

        _obj = OrganizationDataItem.construct(**{
            "id": obj.get("id"),
            "name": obj.get("name"),
            "bucket_id": obj.get("bucketId"),
            "bucket_name": obj.get("bucketName"),
            "bucket_path": obj.get("bucketPath"),
            "dataset": obj.get("dataset"),
            "total_file_count": obj.get("totalFileCount"),
            "total_file_size": obj.get("totalFileSize"),
            "created": obj.get("created"),
            "metadata": obj.get("metadata"),
            "files": [OrganizationDataItemFilesInner.from_dict(_item) for _item in obj.get("files")] if obj.get("files") is not None else None
        })
        return _obj
