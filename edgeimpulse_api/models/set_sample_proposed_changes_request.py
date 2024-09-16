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



from pydantic import BaseModel, Field, StrictInt
from edgeimpulse_api.models.sample_proposed_changes import SampleProposedChanges

class SetSampleProposedChangesRequest(BaseModel):
    job_id: StrictInt = Field(..., alias="jobId", description="Job ID of an AI Actions job. This is passed into your job via the --propose-actions argument.")
    proposed_changes: SampleProposedChanges = Field(..., alias="proposedChanges")
    __properties = ["jobId", "proposedChanges"]

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
    def from_json(cls, json_str: str) -> SetSampleProposedChangesRequest:
        """Create an instance of SetSampleProposedChangesRequest from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of proposed_changes
        if self.proposed_changes:
            _dict['proposedChanges'] = self.proposed_changes.to_dict()
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> SetSampleProposedChangesRequest:
        """Create an instance of SetSampleProposedChangesRequest from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return SetSampleProposedChangesRequest.construct(**obj)

        _obj = SetSampleProposedChangesRequest.construct(**{
            "job_id": obj.get("jobId"),
            "proposed_changes": SampleProposedChanges.from_dict(obj.get("proposedChanges")) if obj.get("proposedChanges") is not None else None
        })
        return _obj

