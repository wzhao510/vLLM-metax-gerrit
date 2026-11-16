import subprocess
import time
import requests
import pytest
import yaml
import re
import os
import signal
import json
from datetime import datetime

# --- 用于存储测试结果，方便最后生成汇总表 ---
summary_results = []

# 加载配置
def load_config():
    with open("mtp_models_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["models"]

# ================= 使用 Session Fixture 保证日志必生成 =================
@pytest.fixture(scope="session", autouse=True)
def generate_final_report():
    # yield 之前的部分在整个测试会话开始前执行
    yield
    
    # yield 之后的部分在所有测试结束后 绝对会执行，无论测试成功还是失败、报错
    print("\n" + "="*30 + " MTP 自动化测试汇总报告 " + "="*30)
    header = f"| {'模型名称 (Model)':<22} | {'MTP 方式 (Method)':<18} | {'测试状态':<10} | {'接受率':<10} | {'备注说明 (Details)'}"
    print(header)
    print("-" * 110)
    
    for res in summary_results:
        # 终端简易排版
        print(f"| {res['Model']:<24} | {res['Method']:<19} | {res['Status']:<10} | {res['Acceptance Rate']:<11} | {res['Details']}")
    print("=" * 86 + " 报告结束 " + "=" * 86)

    # 生成带时间戳的本地 Markdown 文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"mtp_test_report_{timestamp}.md"
    
    try:
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(f"# vLLM MTP 自动化测试报告\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("| 模型名称 (Model) | MTP 实现方式 (Method) | 测试状态 (Status) | 接受率 (Accept Rate) | 备注说明 (Details) |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")
            
            for res in summary_results:
                details_clean = res['Details'].replace('|', '/')
                f.write(f"| {res['Model']} | {res['Method']} | **{res['Status']}** | {res['Acceptance Rate']} | {details_clean} |\n")
                
        print(f"\n[✔] 详细汇总报告已保存至本地文件: {os.path.abspath(report_filename)}\n")
    except Exception as e:
        print(f"\n[✘] 写入报告文件失败: {e}\n")


# ================= 管理 vLLM 服务的生命周期 =================
@pytest.fixture(scope="function")
def vllm_server(request):
    model_config = request.param
    gpu_util = str(model_config.get("gpu_memory_utilization", 0.90))
    
    cmd = [
        "vllm", "serve", model_config["model_path"],
        "-tp", str(model_config["tp"]),
        "--trust-remote-code",
        "--gpu-memory-utilization", gpu_util,
        "--max-model-len", str(model_config["max_model_len"]),
        "--max-num-seqs", str(model_config["max_num_seqs"]),
        "--speculative-config", model_config["speculative_config"],
        "--enforce-eager"
    ]
    
    print(f"\n[INFO] 正在启动模型: {model_config['name']} (GPU 利用率: {gpu_util})")
    
    log_file = open(f"vllm_{model_config['name']}.log", "w")
    process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    
    server_ready = False
    for _ in range(360): # 最长等待 1小时 (360*10秒)
        try:
            resp = requests.get("http://127.0.0.1:8000/v1/models", timeout=2)
            if resp.status_code == 200:
                server_ready = True
                print(f"[INFO] 模型 {model_config['name']} 服务已就绪！")
                break
        except requests.exceptions.RequestException:
            time.sleep(10)
            
    # 不直接抛出 Fail，而是将状态打包传递
    fixture_data = {
        "config": model_config,
        "is_ready": server_ready,
        "error_msg": ""
    }

    if not server_ready:
        fixture_data["error_msg"] = f"服务启动超时 (等待了 1 小时)，请检查底层 vllm_{model_config['name']}.log"
        # 安全关闭僵尸进程
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait()
        log_file.close()

    # 将状态交给测试用例处理
    yield fixture_data
    
    # 如果启动成功，测试结束后正常关闭进程
    if server_ready:
        print(f"\n[INFO] 测试完毕，正在关闭模型: {model_config['name']} 释放显存...")
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait()
        except Exception:
            pass
        finally:
            log_file.close()


# ================= 核心测试逻辑 =================
@pytest.mark.parametrize("vllm_server", load_config(), indirect=True)
def test_mtp_function_and_performance(vllm_server):
    # 解析传入的状态数据
    config = vllm_server["config"]
    model_name = config["name"]
    
    try:
        spec_config_dict = json.loads(config["speculative_config"])
        mtp_method = spec_config_dict.get("method")
        
        if not mtp_method:
            if "model" in spec_config_dict:
                mtp_method = "draft"
            elif "prompt_lookup_max" in spec_config_dict:
                 mtp_method = "ngram" # 兼容部分简写配置
            else:
                mtp_method = "Unknown"
    except Exception:
        mtp_method = "Parse_Error"
    
    test_record = {
        "Model": model_name,
        "Method": mtp_method,
        "Status": "PENDING",
        "Acceptance Rate": "N/A",
        "Details": ""
    }
    
    try:
        # 在用例内部检查启动状态。如果没启动成功，直接抛出异常被捕获记录
        if not vllm_server["is_ready"]:
            raise Exception(vllm_server["error_msg"])

        # --- 推理请求预热 ---
        print("\n[INFO] 开始发送推理请求预热 MTP 引擎...")
        payload = {
            "model": config["model_path"],
            "messages": [{"role": "user", "content": config["prompt"]}],
            "max_tokens": 1000,
            "temperature": 0,
            "stream": False
        }
        
        for _ in range(3):
            response = requests.post("http://127.0.0.1:8000/v1/chat/completions", json=payload)
            assert response.status_code == 200, f"API 请求失败: {response.status_code}"
            
        result_text = response.json()["choices"][0]["message"]["content"]
        assert config["expected_keyword"] in result_text, f"回答中缺少关键词 '{config['expected_keyword']}'"
        
        # --- 计算接受率 ---
        print("[INFO] 开始抓取并计算 MTP 接受率...")
        metrics_resp = requests.get("http://127.0.0.1:8000/metrics")
        assert metrics_resp.status_code == 200, "无法获取 Metrics"
        metrics_text = metrics_resp.text
        
        draft_match = re.search(r'vllm:spec_decode_num_draft_tokens_total\{.*?\}\s+([0-9.]+)', metrics_text)
        accepted_match = re.search(r'vllm:spec_decode_num_accepted_tokens_total\{.*?\}\s+([0-9.]+)', metrics_text)
        
        assert draft_match and accepted_match, "未找到投机解码的 counters 指标！"
        
        draft_tokens = float(draft_match.group(1))
        accepted_tokens = float(accepted_match.group(1))
        assert draft_tokens > 0, "Draft tokens 为 0，MTP 未触发！"
        
        acceptance_rate = accepted_tokens / draft_tokens
        rate_str = f"{acceptance_rate * 100:.2f}%"
        test_record["Acceptance Rate"] = rate_str
        print(f"[INFO] 实际 MTP 接受率: {rate_str}")
        
        min_rate = config["min_acceptance_rate"]
        assert acceptance_rate >= min_rate, f"接受率 {rate_str} 低于预期值 {min_rate*100:.2f}%"
        
        test_record["Status"] = "PASS"
        test_record["Details"] = "功能与接受率均正常"

    except AssertionError as e:
        test_record["Status"] = "FAIL"
        test_record["Details"] = str(e)
        pytest.fail(str(e)) # 抛给 pytest 框架进行标准统计
    except Exception as e:
        test_record["Status"] = "ERROR"
        test_record["Details"] = f"异常: {str(e)}"
        pytest.fail(str(e))
    finally:
        # 无论成功、失败还是 OOM 崩溃，这里都会把结果收录，供最终报告打印
        summary_results.append(test_record)