from server import app


def test_health_route():
    client = app.test_client()
    response = client.get('/api/health')
    assert response.status_code == 200
    assert response.get_json()['status'] == 'healthy'


def test_index_route_serves_debug_ui():
    client = app.test_client()
    response = client.get('/')
    assert response.status_code == 200
    assert b'VocalFusion Prototype Debug UI' in response.data
