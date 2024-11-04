# Example of a simple script to analyze network traffic for unusual patterns (Botnet Detection)
import pandas as pd
from sklearn.ensemble import IsolationForest

# Sample network traffic data (in reality, this would be more complex)
data = pd.DataFrame({
    'source_ip': ['192.168.1.1', '192.168.1.2', '10.0.0.1', '10.0.0.2'],
    'dest_ip': ['10.0.0.3', '10.0.0.4', '192.168.1.3', '192.168.1.4'],
    'bytes': [500, 1500, 3000, 100]
})

# Using Isolation Forest for anomaly detection
clf = IsolationForest(random_state=42)
clf.fit(data[['bytes']])
data['anomaly'] = clf.predict(data[['bytes']])

print(data)
