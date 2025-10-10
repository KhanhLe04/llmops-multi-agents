#!/usr/bin/env python3
"""
Test script cho RAG Agent với LangGraph
Test các chức năng cơ bản và workflow
"""

import os
import sys
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any

# Add parent directory to path để import agent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import RAGAgent, RAGState
from config import Config

class RAGAgentTester:
    def __init__(self):
        """Khởi tạo tester"""
        print("🧪 Khởi tạo RAG Agent Tester...")
        self.agent = None
        self.test_results = []
        
    def initialize_agent(self):
        """Khởi tạo RAG Agent"""
        try:
            print("🤖 Đang khởi tạo RAG Agent...")
            self.agent = RAGAgent()
            print("✅ RAG Agent khởi tạo thành công!")
            return True
        except Exception as e:
            print(f"❌ Lỗi khởi tạo RAG Agent: {e}")
            return False
    
    def test_basic_functionality(self):
        """Test các chức năng cơ bản"""
        print("\n🔍 Testing Basic Functionality...")
        
        # Test 1: Kiểm tra components
        print("📋 Test 1: Kiểm tra components")
        try:
            assert self.agent.llm is not None, "LLM không được khởi tạo"
            assert self.agent.embeddings is not None, "Embeddings không được khởi tạo"
            assert self.agent.qdrant_client is not None, "Qdrant client không được khởi tạo"
            assert self.agent.vector_store is not None, "Vector store không được khởi tạo"
            assert self.agent.workflow is not None, "Workflow không được khởi tạo"
            print("✅ Tất cả components đã được khởi tạo")
        except Exception as e:
            print(f"❌ Lỗi kiểm tra components: {e}")
            return False
        
        # Test 2: Test embedding generation
        print("📋 Test 2: Test embedding generation")
        try:
            test_text = "Xin chào, tôi muốn hỏi về vấn đề stress học tập"
            embedding = self.agent.embeddings.embed_query(test_text)
            assert len(embedding) > 0, "Embedding không được tạo"
            print(f"✅ Embedding generated: dimension={len(embedding)}")
        except Exception as e:
            print(f"❌ Lỗi tạo embedding: {e}")
            return False
        
        # Test 3: Test Qdrant connection
        print("📋 Test 3: Test Qdrant connection")
        try:
            collections = self.agent.qdrant_client.get_collections()
            print(f"✅ Qdrant connected: {len(collections.collections)} collections")
            
            # Check if our collection exists
            collection_names = [col.name for col in collections.collections]
            if Config.COLLECTION_NAME in collection_names:
                print(f"✅ Collection '{Config.COLLECTION_NAME}' exists")
            else:
                print(f"⚠️ Collection '{Config.COLLECTION_NAME}' not found")
                print(f"Available collections: {collection_names}")
        except Exception as e:
            print(f"❌ Lỗi kết nối Qdrant: {e}")
            return False
        
        return True
    
    def test_workflow_nodes(self):
        """Test từng node trong workflow"""
        print("\n🔄 Testing Workflow Nodes...")
        
        # Create test state
        test_state: RAGState = {
            "query": "Tôi đang bị stress vì thi cử, phải làm sao?",
            "user_context": {},
            "query_embedding": None,
            "retrieved_documents": [],
            "relevant_documents": [],
            "context": "",
            "answer": "",
            "sources": [],
            "messages": [],
            "step": "initialized",
            "processing_time": 0.0,
            "status": "initialized",
            "error": None
        }
        
        # Test 1: retrieve_documents_node
        print("📋 Test 1: retrieve_documents_node")
        try:
            result_state = self.agent.retrieve_documents_node(test_state)
            assert "retrieved_documents" in result_state, "retrieved_documents không có trong state"
            assert result_state["status"] in ["document_retrieved", "error"], f"Status không hợp lệ: {result_state['status']}"
            print(f"✅ retrieve_documents_node: {len(result_state['retrieved_documents'])} documents retrieved")
            
            # Update test_state với kết quả
            test_state.update(result_state)
        except Exception as e:
            print(f"❌ Lỗi retrieve_documents_node: {e}")
            return False
        
        # Test 2: filter_documents_node (chỉ test nếu có documents)
        if test_state["retrieved_documents"]:
            print("📋 Test 2: filter_documents_node")
            try:
                result_state = self.agent.filter_documents_node(test_state)
                assert "relevant_documents" in result_state, "relevant_documents không có trong state"
                assert result_state["status"] in ["filtered_documents", "error"], f"Status không hợp lệ: {result_state['status']}"
                print(f"✅ filter_documents_node: {len(result_state['relevant_documents'])} documents filtered")
                
                test_state.update(result_state)
            except Exception as e:
                print(f"❌ Lỗi filter_documents_node: {e}")
                return False
        else:
            print("⚠️ Skip filter_documents_node test - no documents retrieved")
        
        # Test 3: aggregate_context_node
        print("📋 Test 3: aggregate_context_node")
        try:
            result_state = self.agent.aggregate_context_node(test_state)
            assert "context" in result_state, "context không có trong state"
            assert result_state["status"] in ["context_aggregated", "error"], f"Status không hợp lệ: {result_state['status']}"
            print(f"✅ aggregate_context_node: context length={len(result_state['context'])}")
            
            test_state.update(result_state)
        except Exception as e:
            print(f"❌ Lỗi aggregate_context_node: {e}")
            return False
        
        # Test 4: generate_answer_node
        print("📋 Test 4: generate_answer_node")
        try:
            result_state = self.agent.generate_answer_node(test_state)
            assert "answer" in result_state, "answer không có trong state"
            assert result_state["status"] in ["completed", "error"], f"Status không hợp lệ: {result_state['status']}"
            print(f"✅ generate_answer_node: answer length={len(result_state['answer'])}")
            
            test_state.update(result_state)
        except Exception as e:
            print(f"❌ Lỗi generate_answer_node: {e}")
            return False
        
        return True
    
    def test_full_workflow(self):
        """Test toàn bộ workflow"""
        print("\n🚀 Testing Full Workflow...")
        
        test_queries = [
            "Tôi đang bị stress vì thi cử, phải làm sao?",
            "Làm thế nào để quản lý cảm xúc tốt hơn?",
            "Tôi cảm thấy cô đơn ở trường, có ai giúp tôi không?",
            "Phương pháp thư giãn nào hiệu quả cho học sinh?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Test Query {i}: {query}")
            try:
                start_time = datetime.now()
                result = self.agent.invoke(query)
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Validate result structure
                required_fields = ["answer", "sources", "relevant_documents_count", "total_retrieved_count", "processing_time", "status"]
                for field in required_fields:
                    assert field in result, f"Missing field: {field}"
                
                # Print results
                print(f"✅ Status: {result['status']}")
                print(f"✅ Answer length: {len(result['answer'])}")
                print(f"✅ Sources: {result['sources']}")
                print(f"✅ Retrieved: {result['total_retrieved_count']}, Relevant: {result['relevant_documents_count']}")
                print(f"✅ Processing time: {processing_time:.2f}s")
                
                # Save result
                self.test_results.append({
                    "query": query,
                    "result": result,
                    "processing_time": processing_time,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Print answer preview
                answer_preview = result['answer'][:200] + "..." if len(result['answer']) > 200 else result['answer']
                print(f"📄 Answer preview: {answer_preview}")
                
            except Exception as e:
                print(f"❌ Lỗi test query {i}: {e}")
                continue
        
        return len(self.test_results) > 0
    
    def test_error_handling(self):
        """Test error handling"""
        print("\n⚠️ Testing Error Handling...")
        
        # Test với empty query
        print("📋 Test 1: Empty query")
        try:
            result = self.agent.invoke("")
            print(f"✅ Empty query handled: status={result['status']}")
        except Exception as e:
            print(f"❌ Empty query error: {e}")
        
        # Test với very long query
        print("📋 Test 2: Very long query")
        try:
            long_query = "stress " * 1000  # Very long query
            result = self.agent.invoke(long_query)
            print(f"✅ Long query handled: status={result['status']}")
        except Exception as e:
            print(f"❌ Long query error: {e}")
        
        return True
    
    def save_test_results(self):
        """Lưu kết quả test"""
        if not self.test_results:
            print("⚠️ Không có kết quả test để lưu")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, ensure_ascii=False, indent=2)
            print(f"💾 Test results saved to: {filename}")
        except Exception as e:
            print(f"❌ Lỗi lưu kết quả test: {e}")
    
    def run_all_tests(self):
        """Chạy tất cả tests"""
        print("🧪 Bắt đầu test RAG Agent...")
        print("="*60)
        
        # Initialize agent
        if not self.initialize_agent():
            print("❌ Không thể khởi tạo agent, dừng test")
            return False
        
        # Run tests
        tests = [
            ("Basic Functionality", self.test_basic_functionality),
            ("Workflow Nodes", self.test_workflow_nodes),
            ("Full Workflow", self.test_full_workflow),
            ("Error Handling", self.test_error_handling)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n{'='*20} {test_name} {'='*20}")
            try:
                if test_func():
                    print(f"✅ {test_name} PASSED")
                    passed_tests += 1
                else:
                    print(f"❌ {test_name} FAILED")
            except Exception as e:
                print(f"❌ {test_name} ERROR: {e}")
        
        # Summary
        print("\n" + "="*60)
        print(f"📊 TEST SUMMARY: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 Tất cả tests PASSED! RAG Agent hoạt động tốt.")
        else:
            print("⚠️ Một số tests FAILED. Kiểm tra lại implementation.")
        
        # Save results
        self.save_test_results()
        
        return passed_tests == total_tests

def main():
    """Main function"""
    print("🚀 RAG Agent Test Suite")
    print("="*60)
    
    # Check environment
    print("🔍 Checking environment...")
    required_vars = ["GOOGLE_API_KEY", "QDRANT_URL", "COLLECTION_NAME", "EMBEDDING_MODEL"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        print("💡 Please set these variables in your .env file")
        return False
    
    print("✅ Environment variables OK")
    
    # Run tests
    tester = RAGAgentTester()
    success = tester.run_all_tests()
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
