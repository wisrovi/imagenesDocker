Testing
=======

A Python test script (``tests/test.py``) is included to verify the setup:

.. code-block:: bash

   python tests/test.py

This script:

- Sends a test prompt to the Ollama API
- Uses the specified model (default: llama3.1:8b)
- Configures context window and temperature
- Prints the generated response

Modify ``tests/test.py`` variables as needed:

- ``model_name``: Change the model to test
- ``ollama_host``: Update the host URL if necessary
- ``num_ctx``: Adjust context window size
- ``temperature``: Set creativity level