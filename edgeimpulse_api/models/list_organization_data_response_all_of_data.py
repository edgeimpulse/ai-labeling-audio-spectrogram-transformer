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
from typing import Dict
from pydantic import BaseModel, Field, StrictInt, StrictStr

class ListOrganizationDataResponseAllOfData(BaseModel):
    id: StrictInt = ...
    name: StrictStr = ...
    bucket_id: StrictInt = Field(..., alias="bucketId")
    bucket_name: StrictStr = Field(..., alias="bucketName")
    bucket_path: StrictStr = Field(..., alias="bucketPath")
    full_bucket_path: StrictStr = Field(..., alias="fullBucketPath")
    dataset: StrictStr = ...
    total_file_count: StrictInt = Field(..., alias="totalFileCount")
    total_file_size: StrictInt = Field(..., alias="totalFileSize")
    created: datetime = ...
    metadata: Dict[str, StrictStr] = ...
    metadata_string_for_cli: StrictStr = Field(..., alias="metadataStringForCLI", description="String that's passed in to a transformation block in `--metadata` (the metadata + a `dataItemInfo` object)")
    __properties = ["id", "name", "bucketId", "bucketName", "bucketPath", "fullBucketPath", "dataset", "totalFileCount", "totalFileSize", "created", "metadata", "metadataStringForCLI"]

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
    def from_json(cls, json_str: str) -> ListOrganizationDataResponseAllOfData:
        """Create an instance of ListOrganizationDataResponseAllOfData from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> ListOrganizationDataResponseAllOfData:
        """Create an instance of ListOrganizationDataResponseAllOfData from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return ListOrganizationDataResponseAllOfData.construct(**obj)

        _obj = ListOrganizationDataResponseAllOfData.construct(**{
            "id": obj.get("id"),
            "name": obj.get("name"),
            "bucket_id": obj.get("bucketId"),
            "bucket_name": obj.get("bucketName"),
            "bucket_path": obj.get("bucketPath"),
            "full_bucket_path": obj.get("fullBucketPath"),
            "dataset": obj.get("dataset"),
            "total_file_count": obj.get("totalFileCount"),
            "total_file_size": obj.get("totalFileSize"),
            "created": obj.get("created"),
            "metadata": obj.get("metadata"),
            "metadata_string_for_cli": obj.get("metadataStringForCLI")
        })
        return _obj

