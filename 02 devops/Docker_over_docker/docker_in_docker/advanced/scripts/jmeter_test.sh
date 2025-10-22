#!/bin/bash

# JMeter Performance Test Script for Docker-in-Docker project

set -e

echo "⚡ Running JMeter performance tests..."

JMETER_VERSION="5.6.3"
JMETER_HOME="/tmp/jmeter"

# Download and setup JMeter if not present
if [ ! -d "$JMETER_HOME" ]; then
    echo "📥 Downloading JMeter $JMETER_VERSION..."
    wget -q -O /tmp/jmeter.tgz "https://downloads.apache.org/jmeter/binaries/apache-jmeter-${JMETER_VERSION}.tgz"
    tar -xzf /tmp/jmeter.tgz -C /tmp
    mv "/tmp/apache-jmeter-${JMETER_VERSION}" "$JMETER_HOME"
fi

# Set JMeter path
export PATH="$JMETER_HOME/bin:$PATH"

# Run the test
echo "🏃 Running performance test..."
jmeter -n \
    -t test/performance_test.jmx \
    -l test/results.jtl \
    -j test/jmeter.log \
    -e -o test/report

# Generate summary report
echo "📊 Generating performance report..."
echo "=== Performance Test Results ==="
echo "Average Response Time: $(grep -o '"avg": [0-9]*' test/report/statistics.json | head -1 | cut -d' ' -f2)"
echo "95th Percentile: $(grep -o '"pct95": [0-9]*' test/report/statistics.json | head -1 | cut -d' ' -f2)"
echo "Error Rate: $(grep -o '"ko": [0-9]*' test/report/statistics.json | head -1 | cut -d' ' -f2)"

# Check thresholds
AVG_RESPONSE=$(grep -o '"avg": [0-9]*' test/report/statistics.json | head -1 | cut -d' ' -f2 | tr -d '"')
ERROR_RATE=$(grep -o '"ko": [0-9]*' test/report/statistics.json | head -1 | cut -d' ' -f2 | tr -d '"')

if [ "$AVG_RESPONSE" -gt 1000 ]; then
    echo "⚠️  Average response time is high: ${AVG_RESPONSE}ms"
    exit 1
fi

if [ "$ERROR_RATE" -gt 5 ]; then
    echo "❌ Error rate is too high: ${ERROR_RATE}%"
    exit 1
fi

echo "✅ Performance test passed!"
echo "📈 Full report available at: test/report/index.html"