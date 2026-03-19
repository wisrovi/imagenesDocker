import pytest
from unittest.mock import MagicMock, patch
from pr_analyzer.github_client import GitHubClient

# Pytest fixture to set up a reusable GitHubClient instance for tests.
# Fixtures are a powerful feature of pytest for creating test environments.
@pytest.fixture
def mock_github_client():
    """ 
    This fixture creates an instance of the GitHubClient for testing.
    It uses a patch to mock the `_get_github_token` method, so no real
    environment variable or `secrets.env` file is needed for tests.
    This ensures that tests are isolated and don't depend on external configuration.
    """
    # The `with` statement ensures the patch is active only during the fixture's scope.
    with patch('pr_analyzer.github_client.GitHubClient._get_github_token', return_value='fake_token'):
        client = GitHubClient(owner='test_owner', repo='test_repo')
        yield client # yield is used in fixtures to provide the object to the test function

def test_get_open_pull_requests_success(mock_github_client, mocker):
    """
    Tests the `get_open_pull_requests` method for a successful API call.
    
    Validates:
    - That the method correctly parses the JSON response from the API.
    - That it returns a list containing the pull request data.
    """
    # Create a mock object to simulate the response from the `requests.get` call.
    mock_response = MagicMock()
    # Define the JSON payload that the mock API will return.
    mock_response.json.return_value = [{'number': 1, 'title': 'Test PR'}]
    # `raise_for_status` should do nothing in a successful case.
    mock_response.raise_for_status.return_value = None
    
    # Use `mocker.patch` to replace `requests.get` with our mock response.
    # This intercepts the real HTTP call and returns our predefined data instead.
    mocker.patch('requests.get', return_value=mock_response)

    # Call the method under test.
    prs = mock_github_client.get_open_pull_requests()

    # Assertions: Verify that the output is what we expect.
    # Check if one PR was returned.
    assert len(prs) == 1
    # Check if the title of the returned PR is correct.
    assert prs[0]['title'] == 'Test PR'

def test_get_pr_details_success(mock_github_client, mocker):
    """
    Tests the `get_pr_details` method for a successful API call.
    This method makes two separate API calls, so we must mock both.

    Validates:
    - That the review data is correctly aggregated (approved count, changes requested).
    - That the file list is correctly parsed.
    """
    # --- Mock for the first API call (reviews) ---
    mock_reviews_response = MagicMock()
    mock_reviews_response.json.return_value = [
        {'state': 'APPROVED'}, # One approval
        {'state': 'CHANGES_REQUESTED'} # One request for changes
    ]
    mock_reviews_response.raise_for_status.return_value = None

    # --- Mock for the second API call (files) ---
    mock_files_response = MagicMock()
    mock_files_response.json.return_value = [
        {'filename': 'src/main.py'}
    ]
    mock_files_response.raise_for_status.return_value = None

    # `requests.get` will be called twice. `side_effect` allows us to return
    # different values for each call.
    mocker.patch('requests.get', side_effect=[mock_reviews_response, mock_files_response])

    # Call the method under test.
    details = mock_github_client.get_pr_details(pr_number=1)

    # Assertions: Verify the aggregated results.
    # Check that the single 'APPROVED' state was counted.
    assert details['approved_count'] == 1
    # Check that the 'CHANGES_REQUESTED' state was detected.
    assert details['has_changes_requested'] is True
    # Check that the file list was correctly retrieved.
    assert len(details['files']) == 1
    assert details['files'][0]['filename'] == 'src/main.py'