#!/usr/bin/env python3
"""
Load testing script for LegalAdvisor API
Sử dụng locust để test hiệu suất dưới tải cao
"""

from locust import HttpUser, task, between, TaskSet
import random
import json

# Các câu hỏi test
TEST_QUESTIONS = [
    "Luật lao động có quy định gì về giờ làm việc?",
    "Thế nào là hợp đồng lao động?",
    "Quyền và nghĩa vụ của người lao động là gì?",
    "Công ty có quyền sa thải người lao động không?",
    "Lương tối thiểu vùng là bao nhiêu?",
    "Bảo hiểm xã hội bắt buộc hay tự nguyện?",
    "Thời gian thử việc tối đa bao lâu?",
    "Phúc lợi của công nhân bao gồm những gì?",
    "Quy định về ngày lễ công khai?",
    "Thủ tục giải quyết tranh chấp lao động?",
]

class UserBehavior(TaskSet):
    """Định nghĩa các hành vi của user"""

    @task(3)
    def ask_question(self):
        """Test /ask endpoint"""
        question = random.choice(TEST_QUESTIONS)
        
        response = self.client.post(
            "/ask",
            json={"question": question},
            catch_response=True
        )
        
        if response.status_code == 200:
            response.success()
        elif response.status_code == 429:  # Rate limited
            print(f"⚠️ Rate limited: {response.text}")
            response.success()  # Expected behavior
        else:
            response.failure(f"Unexpected status code: {response.status_code}")

    @task(1)
    def health_check(self):
        """Test /health endpoint"""
        response = self.client.get(
            "/health",
            catch_response=True
        )
        
        if response.status_code == 200:
            response.success()
        else:
            response.failure(f"Health check failed: {response.status_code}")

    @task(1)
    def get_sources(self):
        """Test /sources endpoint"""
        chunk_id = random.randint(1, 61425)
        
        response = self.client.get(
            f"/sources/{chunk_id}",
            catch_response=True
        )
        
        if response.status_code in [200, 404]:  # 404 OK for invalid ID
            response.success()
        else:
            response.failure(f"Unexpected status code: {response.status_code}")


class LegalAdvisorUser(HttpUser):
    """Định nghĩa user cho load test"""
    
    tasks = [UserBehavior]
    wait_time = between(1, 3)  # Thời gian chờ giữa các request (1-3s)


# Run configurations:
# 1. GUI mode (mặc định):
#    locust -f tests/load_test.py --host=http://localhost:8000
#
# 2. Headless mode (không GUI):
#    locust -f tests/load_test.py --host=http://localhost:8000 \
#      --users 100 --spawn-rate 10 --run-time 5m --headless
#
# 3. Chi tiết ngành hàng (CSV):
#    locust -f tests/load_test.py --host=http://localhost:8000 \
#      --csv=results --users 100 --run-time 5m --headless
#
# Expected Results:
# - Response time p50: < 500ms
# - Response time p99: < 5000ms
# - Failure rate: < 1%
# - Rate limiting: Khoảng 70% bị limited khi vượt ngưỡng
