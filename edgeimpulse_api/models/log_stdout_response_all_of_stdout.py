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
from typing import Optional
from pydantic import BaseModel, Field, StrictStr, validator

class LogStdoutResponseAllOfStdout(BaseModel):
    created: datetime = ...
    data: StrictStr = ...
    log_level: Optional[StrictStr] = Field(None, alias="logLevel")
    __properties = ["created", "data", "logLevel"]

    @validator('log_level')
    def log_level_validate_enum(cls, v):
        if v is None:
            return v

        if v not in ('error', 'warn', 'info', 'debug'):
            raise ValueError("must validate the enum values ('error', 'warn', 'info', 'debug')")
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
    def from_json(cls, json_str: str) -> LogStdoutResponseAllOfStdout:
        """Create an instance of LogStdoutResponseAllOfStdout from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> LogStdoutResponseAllOfStdout:
        """Create an instance of LogStdoutResponseAllOfStdout from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return LogStdoutResponseAllOfStdout.construct(**obj)

        _obj = LogStdoutResponseAllOfStdout.construct(**{
            "created": obj.get("created"),
            "data": obj.get("data"),
            "log_level": obj.get("logLevel")
        })
        return _obj

