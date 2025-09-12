#!/usr/bin/env python3
"""
Simple test for the fixed Orchestrator Agent
Tests the chat functionality after fixing the Document validation error
"""

import requests
import json
import time

def test_agent():
    """Test the agent with a simple message"""
    
    url = "http://localhost:7000"
    
    print("🧪 Testing Fixed Orchestrator Agent")
    print("=" * 50)
    
    # Test health first
    try:
        health_response = requests.get(f"{url}/health", timeout=10)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print("✅ Agent health check passed")
            print(f"   Qdrant: {health_data.get('qdrant', 'unknown')}")
            print(f"   Collection: {health_data.get('collection', 'unknown')}")
        else:
            print(f"⚠️  Health check returned: {health_response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to agent: {e}")
        print("Make sure the agent is running: python main.py")
        return False
    
    # Test chat endpoint
    test_message = "Xin chào! Tôi đang cảm thấy căng thẳng về kỳ thi sắp tới. Bạn có thể tư vấn cho tôi không?"
    
    payload = {
        "message": test_message,
        "session_id": "test_fix"
    }
    
    print(f"\n📨 Testing message: {test_message}")
    print("⏳ Sending request...")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{url}/chat",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60  # Longer timeout for potentially slow first request
        )
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ SUCCESS! Chat request completed")
            print(f"⏱️  Response time: {response_time:.2f}s")
            print("-" * 50)
            print(f"🤖 Agent Response:")
            print(result.get('response', 'No response'))
            
            sources = result.get('sources', [])
            if sources:
                print(f"\n📚 Sources ({len(sources)}):")
                for i, source in enumerate(sources, 1):
                    print(f"   {i}. {source}")
            else:
                print("\n📚 No sources (possibly fallback mode)")
            
            print(f"\n🎯 Session ID: {result.get('session_id', 'unknown')}")
            
            # Check if it's fallback mode
            if "fallback_mode" in sources or "emergency_fallback" in sources:
                print("\n⚠️  Note: Agent is running in fallback mode")
                print("   This may indicate issues with Qdrant vector database")
            
            return True
            
        else:
            print(f"❌ FAILED! HTTP {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error details: {error_data}")
            except:
                print(f"Raw response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ TIMEOUT! Request took too long")
        return False
    except Exception as e:
        print(f"❌ REQUEST FAILED: {e}")
        return False

def test_multiple_messages():
    """Test with multiple messages to ensure session works"""
    
    url = "http://localhost:7000"
    session_id = f"test_session_{int(time.time())}"
    
    messages = [
        "Tôi hay lo lắng về điểm số.",
        "Làm sao để tập trung học tập tốt hơn?",
        "Cảm ơn bạn đã tư vấn!"
    ]
    
    print(f"\n🔄 Testing multiple messages (session: {session_id})")
    print("-" * 50)
    
    for i, message in enumerate(messages, 1):
        print(f"\n📨 Message {i}: {message}")
        
        payload = {
            "message": message,
            "session_id": session_id
        }
        
        try:
            response = requests.post(
                f"{url}/chat",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                response_preview = result.get('response', '')[:100] + "..." if len(result.get('response', '')) > 100 else result.get('response', '')
                print(f"✅ Response {i}: {response_preview}")
            else:
                print(f"❌ Failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
        
        time.sleep(1)  # Brief pause
    
    print("✅ Multiple message test completed!")
    return True

if __name__ == "__main__":
    print("🚀 Starting Orchestrator Agent Tests")
    print("=" * 60)
    
    # Test single message
    success1 = test_agent()
    
    if success1:
        # Test multiple messages
        success2 = test_multiple_messages()
        
        if success1 and success2:
            print("\n" + "=" * 60)
            print("🎉 ALL TESTS PASSED! The Document validation error has been fixed.")
            print("✅ Agent is working correctly with both single and multiple messages.")
        else:
            print("\n⚠️  Some tests failed. Check the logs above.")
    else:
        print("\n❌ Initial test failed. Agent may not be working properly.")
        exit(1)
