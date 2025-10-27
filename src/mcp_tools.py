"""
MCP (Model Context Protocol) 工具调用接口
支持关键词检索和BM25+BGE-M3稠密检索
"""

import json
import re
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """检索结果数据结构"""
    content: str
    score: float
    source: str
    metadata: Dict[str, Any]


class RetrievalTool(ABC):
    """检索工具基类"""
    
    @abstractmethod
    def search(self, query: str, table_data: Dict, top_k: int = 5) -> List[RetrievalResult]:
        """执行检索"""
        pass


class KeywordRetrieval(RetrievalTool):
    """关键词稀疏检索"""
    
    def __init__(self):
        self.name = "keyword_retrieval"
    
    def search(self, query: str, table_data: Dict, top_k: int = 5) -> List[RetrievalResult]:
        """基于关键词的稀疏检索"""
        
        # 提取查询关键词
        keywords = self._extract_keywords(query)
        
        # 在表格中搜索匹配
        results = []
        columns = table_data.get('columns', [])
        rows = table_data.get('data', [])
        
        for i, row in enumerate(rows):
            for j, cell in enumerate(row):
                cell_str = str(cell).lower()
                
                # 计算匹配分数
                score = self._calculate_keyword_score(keywords, cell_str)
                
                if score > 0:
                    results.append(RetrievalResult(
                        content=str(cell),
                        score=score,
                        source=f"row_{i}_col_{j}",
                        metadata={
                            "row": i,
                            "column": columns[j] if j < len(columns) else f"col_{j}",
                            "column_index": j,
                            "keywords_matched": [kw for kw in keywords if kw in cell_str]
                        }
                    ))
        
        # 按分数排序并返回top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def _extract_keywords(self, query: str) -> List[str]:
        """提取查询关键词"""
        # 简单的关键词提取
        words = re.findall(r'\b\w+\b', query.lower())
        
        # 过滤停用词
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords
    
    def _calculate_keyword_score(self, keywords: List[str], text: str) -> float:
        """计算关键词匹配分数"""
        if not keywords:
            return 0.0
        
        matches = sum(1 for keyword in keywords if keyword in text)
        return matches / len(keywords)


class BM25BGERetrieval(RetrievalTool):
    """BM25 + BGE-M3 稠密检索"""
    
    def __init__(self, bge_model_path: str = None):
        self.name = "bm25_bge_retrieval"
        self.bge_model_path = bge_model_path
        self._bge_model = None
        self._bge_tokenizer = None
    
    def _load_bge_model(self):
        """延迟加载BGE模型"""
        if self._bge_model is None:
            try:
                from transformers import AutoTokenizer, AutoModel
                self._bge_tokenizer = AutoTokenizer.from_pretrained(self.bge_model_path)
                self._bge_model = AutoModel.from_pretrained(self.bge_model_path)
                print("✓ BGE模型加载完成")
            except Exception as e:
                print(f"⚠️ BGE模型加载失败: {e}")
                print("将使用简化的稠密检索")
    
    def search(self, query: str, table_data: Dict, top_k: int = 5) -> List[RetrievalResult]:
        """BM25 + BGE稠密检索"""
        
        # 1. BM25稀疏检索
        bm25_results = self._bm25_search(query, table_data)
        
        # 2. BGE稠密检索
        dense_results = self._dense_search(query, table_data)
        
        # 3. 融合结果
        combined_results = self._combine_results(bm25_results, dense_results)
        
        return combined_results[:top_k]
    
    def _bm25_search(self, query: str, table_data: Dict) -> List[RetrievalResult]:
        """BM25稀疏检索"""
        # 简化的BM25实现
        results = []
        columns = table_data.get('columns', [])
        rows = table_data.get('data', [])
        
        # 构建文档
        documents = []
        for i, row in enumerate(rows):
            for j, cell in enumerate(row):
                documents.append({
                    "content": str(cell),
                    "row": i,
                    "column": columns[j] if j < len(columns) else f"col_{j}",
                    "column_index": j
                })
        
        # 计算BM25分数
        for doc in documents:
            score = self._calculate_bm25_score(query, doc["content"])
            if score > 0:
                results.append(RetrievalResult(
                    content=doc["content"],
                    score=score,
                    source=f"bm25_row_{doc['row']}_col_{doc['column_index']}",
                    metadata={
                        "row": doc["row"],
                        "column": doc["column"],
                        "column_index": doc["column_index"],
                        "method": "bm25"
                    }
                ))
        
        return results
    
    def _dense_search(self, query: str, table_data: Dict) -> List[RetrievalResult]:
        """BGE稠密检索"""
        # 加载BGE模型
        self._load_bge_model()
        
        if self._bge_model is None:
            # 如果BGE模型不可用，使用简化的语义匹配
            return self._simple_semantic_search(query, table_data)
        
        try:
            # 编码查询
            query_embedding = self._encode_text(query)
            
            results = []
            columns = table_data.get('columns', [])
            rows = table_data.get('data', [])
            
            for i, row in enumerate(rows):
                for j, cell in enumerate(row):
                    cell_embedding = self._encode_text(str(cell))
                    
                    # 计算余弦相似度
                    similarity = self._cosine_similarity(query_embedding, cell_embedding)
                    
                    if similarity > 0.3:  # 阈值过滤
                        results.append(RetrievalResult(
                            content=str(cell),
                            score=similarity,
                            source=f"bge_row_{i}_col_{j}",
                            metadata={
                                "row": i,
                                "column": columns[j] if j < len(columns) else f"col_{j}",
                                "column_index": j,
                                "method": "bge",
                                "similarity": similarity
                            }
                        ))
            
            return results
            
        except Exception as e:
            print(f"⚠️ BGE检索失败: {e}")
            return self._simple_semantic_search(query, table_data)
    
    def _simple_semantic_search(self, query: str, table_data: Dict) -> List[RetrievalResult]:
        """简化的语义搜索（当BGE不可用时）"""
        results = []
        columns = table_data.get('columns', [])
        rows = table_data.get('data', [])
        
        query_lower = query.lower()
        
        for i, row in enumerate(rows):
            for j, cell in enumerate(row):
                cell_str = str(cell).lower()
                
                # 简单的语义匹配
                score = self._simple_semantic_score(query_lower, cell_str)
                
                if score > 0.2:
                    results.append(RetrievalResult(
                        content=str(cell),
                        score=score,
                        source=f"simple_semantic_row_{i}_col_{j}",
                        metadata={
                            "row": i,
                            "column": columns[j] if j < len(columns) else f"col_{j}",
                            "column_index": j,
                            "method": "simple_semantic"
                        }
                    ))
        
        return results
    
    def _simple_semantic_score(self, query: str, text: str) -> float:
        """简单的语义匹配分数"""
        # 基于词汇重叠的简单匹配
        query_words = set(query.split())
        text_words = set(text.split())
        
        if not query_words or not text_words:
            return 0.0
        
        overlap = len(query_words.intersection(text_words))
        return overlap / len(query_words)
    
    def _encode_text(self, text: str) -> np.ndarray:
        """编码文本为向量"""
        if self._bge_model is None or self._bge_tokenizer is None:
            return np.random.rand(768)  # 随机向量作为fallback
        
        inputs = self._bge_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self._bge_model(**inputs)
            # 使用[CLS] token的embedding
            return outputs.last_hidden_state[:, 0].cpu().numpy().flatten()
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _calculate_bm25_score(self, query: str, document: str) -> float:
        """计算BM25分数（简化版）"""
        # 简化的BM25实现
        query_terms = query.lower().split()
        doc_terms = document.lower().split()
        
        if not query_terms or not doc_terms:
            return 0.0
        
        score = 0.0
        for term in query_terms:
            term_freq = doc_terms.count(term)
            if term_freq > 0:
                # 简化的BM25公式
                score += term_freq / len(doc_terms)
        
        return score
    
    def _combine_results(self, bm25_results: List[RetrievalResult], dense_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """融合BM25和稠密检索结果"""
        # 简单的融合策略：加权平均
        combined = {}
        
        # 处理BM25结果
        for result in bm25_results:
            key = f"{result.metadata['row']}_{result.metadata['column_index']}"
            combined[key] = {
                "content": result.content,
                "bm25_score": result.score,
                "bge_score": 0.0,
                "metadata": result.metadata
            }
        
        # 处理BGE结果
        for result in dense_results:
            key = f"{result.metadata['row']}_{result.metadata['column_index']}"
            if key in combined:
                combined[key]["bge_score"] = result.score
            else:
                combined[key] = {
                    "content": result.content,
                    "bm25_score": 0.0,
                    "bge_score": result.score,
                    "metadata": result.metadata
                }
        
        # 计算融合分数
        final_results = []
        for key, data in combined.items():
            # 加权融合：BM25权重0.3，BGE权重0.7
            combined_score = 0.3 * data["bm25_score"] + 0.7 * data["bge_score"]
            
            final_results.append(RetrievalResult(
                content=data["content"],
                score=combined_score,
                source=f"combined_{key}",
                metadata={
                    **data["metadata"],
                    "bm25_score": data["bm25_score"],
                    "bge_score": data["bge_score"],
                    "combined_score": combined_score
                }
            ))
        
        # 按融合分数排序
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results


class MCPToolManager:
    """MCP工具管理器"""
    
    def __init__(self):
        self.tools = {
            "keyword": KeywordRetrieval(),
            "bm25_bge": BM25BGERetrieval()
        }
    
    def get_tool(self, tool_name: str) -> Optional[RetrievalTool]:
        """获取指定工具"""
        return self.tools.get(tool_name)
    
    def list_tools(self) -> List[str]:
        """列出可用工具"""
        return list(self.tools.keys())
    
    def search(self, tool_name: str, query: str, table_data: Dict, top_k: int = 5) -> List[RetrievalResult]:
        """使用指定工具执行检索"""
        tool = self.get_tool(tool_name)
        if tool is None:
            raise ValueError(f"工具不存在: {tool_name}")
        
        return tool.search(query, table_data, top_k)
    
    def multi_tool_search(self, query: str, table_data: Dict, top_k: int = 5) -> Dict[str, List[RetrievalResult]]:
        """使用多个工具执行检索"""
        results = {}
        
        for tool_name, tool in self.tools.items():
            try:
                results[tool_name] = tool.search(query, table_data, top_k)
            except Exception as e:
                print(f"⚠️ 工具 {tool_name} 执行失败: {e}")
                results[tool_name] = []
        
        return results


