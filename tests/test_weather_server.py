import os

# import after setting environment variable
os.environ['OPENWEATHER_API_KEY'] = 'dummy'

import mcp_playground.weather_server as ws


# import after setting environment variable

def test_get_weather_success(monkeypatch):
    class FakeResp:
        status_code = 200

        def json(self):
            return {
                'weather': [{'description': 'clear sky'}],
                'main': {'temp': 20, 'feels_like': 19, 'humidity': 50},
                'wind': {'speed': 1.5},
                'name': 'London',
            }

        def raise_for_status(self):
            pass

    monkeypatch.setattr(ws.requests, 'get', lambda *a, **k: FakeResp())
    result = ws.get_weather('London')
    assert result['location'] == 'London'
    assert 'temperature_celsius' in result


def test_get_weather_network_error(monkeypatch):
    def raise_exc(*args, **kwargs):
        raise ws.requests.exceptions.RequestException('fail')

    monkeypatch.setattr(ws.requests, 'get', raise_exc)
    result = ws.get_weather('London')
    assert 'error' in result