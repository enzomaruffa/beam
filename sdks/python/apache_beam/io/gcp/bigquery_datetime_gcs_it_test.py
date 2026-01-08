#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Integration test for BigQuery DATETIME with Storage Write API using GCS source.

This test uses ReadFromText instead of beam.Create() to work around
Impulse triggering issues with Storage Write API on Dataflow.
"""

import json
import logging
import os
import secrets
import time
import unittest

import hamcrest as hc
import pytest

import apache_beam as beam
from apache_beam.io.gcp.bigquery import WriteToBigQuery
from apache_beam.io.gcp.bigquery_tools import BigQueryWrapper
from apache_beam.io.gcp.internal.clients import bigquery
from apache_beam.io.gcp.tests.bigquery_matcher import BigqueryFullResultMatcher
from apache_beam.testing.test_pipeline import TestPipeline

try:
  from apitools.base.py.exceptions import HttpError
except ImportError:
  HttpError = None

_LOGGER = logging.getLogger(__name__)

# GCS path to test data file
GCS_TEST_DATA = 'gs://dataflow-temp-deployer/test/datetime_test.json'


@unittest.skipIf(HttpError is None, 'GCP dependencies are not installed')
class BigQueryDatetimeGcsIntegrationTests(unittest.TestCase):
  """Integration tests for BigQuery DATETIME using GCS file source."""

  BIG_QUERY_DATASET_ID = 'python_datetime_gcs_it_test_'

  def setUp(self):
    self.test_pipeline = TestPipeline(is_integration_test=True)
    self.runner_name = type(self.test_pipeline.runner).__name__
    self.project = self.test_pipeline.get_option('project')

    self.bigquery_client = BigQueryWrapper()
    self.dataset_id = '%s%d%s' % (
        self.BIG_QUERY_DATASET_ID, int(time.time()), secrets.token_hex(3))
    self.bigquery_client.get_or_create_dataset(self.project, self.dataset_id)
    _LOGGER.info(
        "Created dataset %s in project %s", self.dataset_id, self.project)

  def tearDown(self):
    request = bigquery.BigqueryDatasetsDeleteRequest(
        projectId=self.project, datasetId=self.dataset_id, deleteContents=True)
    try:
      _LOGGER.info(
          "Deleting dataset %s in project %s", self.dataset_id, self.project)
      self.bigquery_client.client.datasets.Delete(request)
    except HttpError:
      _LOGGER.debug(
          'Failed to clean up dataset %s in project %s',
          self.dataset_id,
          self.project)

  @pytest.mark.uses_gcp_java_expansion_service
  @unittest.skipUnless(
      os.environ.get('EXPANSION_JARS'),
      "EXPANSION_JARS environment var is not provided, "
      "indicating that jars have not been built")
  def test_datetime_storage_write_api_from_gcs(self):
    """Test DATETIME with Storage Write API using GCS file source."""
    table_name = 'datetime_storage_write_gcs'
    table_id = '{}.{}'.format(self.dataset_id, table_name)

    table_schema = {
        "fields": [{
            "name": "id", "type": "INTEGER", "mode": "REQUIRED"
        }, {
            "name": "event_time", "type": "DATETIME", "mode": "REQUIRED"
        },
                   {
                       "name": "optional_time",
                       "type": "DATETIME",
                       "mode": "NULLABLE"
                   }]
    }

    expected_data = [(1, '2021-01-15T10:30:00', '2021-01-15T11:30:00'),
                     (2, '2021-06-15T12:00:00', None)]

    pipeline_verifiers = [
        BigqueryFullResultMatcher(
            project=self.project,
            query=(
                "SELECT id, event_time, optional_time FROM %s ORDER BY id" %
                table_id),
            data=expected_data)
    ]

    args = self.test_pipeline.get_full_options_as_args()

    with beam.Pipeline(argv=args) as p:
      _ = (
          p
          | 'ReadFromGCS' >> beam.io.ReadFromText(GCS_TEST_DATA)
          | 'ParseJSON' >> beam.Map(lambda line: json.loads(line))
          | 'WriteToBQ' >> WriteToBigQuery(
              table=table_id,
              schema=table_schema,
              method=WriteToBigQuery.Method.STORAGE_WRITE_API))

    hc.assert_that(p, hc.all_of(*pipeline_verifiers))


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  unittest.main()
