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


from typing import List, Optional
from pydantic import BaseModel, Field, StrictBool, StrictInt, StrictStr, validator
from edgeimpulse_api.models.organization_pipeline_step import OrganizationPipelineStep

class OrganizationUpdatePipelineBody(BaseModel):
    name: StrictStr = ...
    description: StrictStr = ...
    interval_str: Optional[StrictStr] = Field(None, alias="intervalStr", description="15m for every 15 minutes, 2h for every 2 hours, 1d for every 1 day")
    steps: List[OrganizationPipelineStep] = ...
    dataset: Optional[StrictStr] = None
    project_id: Optional[StrictInt] = Field(None, alias="projectId")
    email_recipient_uids: List[StrictInt] = Field(..., alias="emailRecipientUids")
    notification_webhook: Optional[StrictStr] = Field(None, alias="notificationWebhook")
    when_to_email: StrictStr = Field(..., alias="whenToEmail")
    archived: Optional[StrictBool] = None
    __properties = ["name", "description", "intervalStr", "steps", "dataset", "projectId", "emailRecipientUids", "notificationWebhook", "whenToEmail", "archived"]

    @validator('when_to_email')
    def when_to_email_validate_enum(cls, v):
        if v not in ('always', 'on_new_data', 'never'):
            raise ValueError("must validate the enum values ('always', 'on_new_data', 'never')")
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
    def from_json(cls, json_str: str) -> OrganizationUpdatePipelineBody:
        """Create an instance of OrganizationUpdatePipelineBody from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of each item in steps (list)
        _items = []
        if self.steps:
            for _item in self.steps:
                if _item:
                    _items.append(_item.to_dict())
            _dict['steps'] = _items
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> OrganizationUpdatePipelineBody:
        """Create an instance of OrganizationUpdatePipelineBody from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return OrganizationUpdatePipelineBody.construct(**obj)

        _obj = OrganizationUpdatePipelineBody.construct(**{
            "name": obj.get("name"),
            "description": obj.get("description"),
            "interval_str": obj.get("intervalStr"),
            "steps": [OrganizationPipelineStep.from_dict(_item) for _item in obj.get("steps")] if obj.get("steps") is not None else None,
            "dataset": obj.get("dataset"),
            "project_id": obj.get("projectId"),
            "email_recipient_uids": obj.get("emailRecipientUids"),
            "notification_webhook": obj.get("notificationWebhook"),
            "when_to_email": obj.get("whenToEmail"),
            "archived": obj.get("archived")
        })
        return _obj
