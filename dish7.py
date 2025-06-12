import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from fastai.vision.all import *
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
import base64
import random
import matplotlib.pyplot as plt
import time
import shutil
import pathlib
import torch
import pickle
import asyncio
import uuid
from matplotlib import font_manager as fm

# 初始化事件循环
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# 设置页面配置
st.set_page_config(page_title="食堂菜品识别系统", layout="wide")

# 全局配置
APP_CONFIG = {
    "model_path": "dish.pkl",
    "dishes_file": "菜品介绍.xlsx",
    "ratings_file": "评分数据.xlsx",
    "backups_dir": "ratings_backups",
    "font_path": "SourceHanSansSC-Regular.otf",
    "cache_timeout": 3600  # 缓存1小时
}

# 初始化会话状态
if 'user_id' not in st.session_state:
    # 使用 st.experimental_get_query_params() 获取URL参数
    url_params = st.experimental_get_query_params()
    user_id = url_params.get("user_id", [str(uuid.uuid4())[:8]])[0]
    st.session_state.user_id = user_id
    
if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = []
    
if 'current_page' not in st.session_state:
    st.session_state.current_page = "首页"
    
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    
if 'collab_model' not in st.session_state:
    st.session_state.collab_model = None

# 创建备份目录
BACKUP_DIR = Path(__file__).parent / APP_CONFIG["backups_dir"]
BACKUP_DIR.mkdir(exist_ok=True)

# --------------------- 缓存数据和模型 ---------------------
@st.cache_resource(ttl=APP_CONFIG["cache_timeout"])
def load_model():
    """加载菜品识别模型（带缓存）"""
    try:
        model_path = pathlib.Path(__file__).parent / APP_CONFIG["model_path"]
        return load_learner(model_path)
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None

@st.cache_data(ttl=APP_CONFIG["cache_timeout"])
def load_dishes_data():
    """加载菜品信息（带缓存）"""
    try:
        dishes_file = pathlib.Path(__file__).parent / APP_CONFIG["dishes_file"]
        return pd.read_excel(dishes_file)
    except Exception as e:
        st.error(f"菜品信息加载失败: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)  # 5分钟刷新一次
def load_all_ratings(all_users=False):
    """加载评分数据（带缓存）"""
    try:
        ratings_file = pathlib.Path(__file__).parent / APP_CONFIG["ratings_file"]
        if ratings_file.exists():
            df = pd.read_excel(ratings_file)
            if all_users:
                return df
            else:
                return df[df['user_id'] == st.session_state.user_id]
        return pd.DataFrame(columns=['user_id', 'dish_id', 'rating', 'timestamp'])
    except Exception as e:
        st.warning(f"评分数据加载失败: {e}")
        return pd.DataFrame(columns=['user_id', 'dish_id', 'rating', 'timestamp'])

# --------------------- 字体配置（优化加载） ---------------------
def setup_fonts():
    """设置字体（静默处理，避免加载提示）"""
    try:
        font_path = pathlib.Path(__file__).parent / APP_CONFIG["font_path"]
        if font_path.exists():
            fm.fontManager.addfont(str(font_path))
            plt.rcParams["font.family"] = ["Source Han Sans SC", "sans-serif"]
            
            with open(font_path, "rb") as f:
                font_bytes = f.read()
            base64_font = base64.b64encode(font_bytes).decode()
            st.markdown(f"""
            <style>
            @font-face {{
                font-family: 'Source Han Sans SC';
                src: url('data:font/opentype;base64,{base64_font}') format('opentype');
            }}
            body, .stButton, .stTextInput, .stMarkdown {{
                font-family: 'Source Han Sans SC', sans-serif !important;
            }}
            </style>
            """, unsafe_allow_html=True)
        else:
            plt.rcParams["font.family"] = ["SimHei", "sans-serif"]
    except:
        plt.rcParams["font.family"] = ["sans-serif"]
    
    plt.rcParams['axes.unicode_minus'] = False

# --------------------- 数据管理函数 ---------------------
def backup_ratings():
    """备份评分数据"""
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = f"{BACKUP_DIR}/ratings_{timestamp}.xlsx"
        ratings_file = pathlib.Path(__file__).parent / APP_CONFIG["ratings_file"]
        if ratings_file.exists():
            shutil.copy2(ratings_file, backup_path)
            return backup_path
        return None
    except Exception as e:
        st.warning(f"备份失败: {e}")
        return None

def save_rating_safely(dish_id, rating):
    """安全保存评分数据"""
    if not (1 <= rating <= 5):
        return False, "评分需在1-5星范围内"
    
    new_rating = pd.DataFrame({
        'user_id': [st.session_state.user_id],
        'dish_id': [dish_id],
        'rating': [rating],
        'timestamp': [pd.Timestamp.now()]
    })
    
    try:
        backup_path = backup_ratings()
        if backup_path:
            st.info(f"已创建评分数据备份: {backup_path}")
            
        # 加载所有用户数据，移除当前用户的旧评分，再添加新评分
        all_data = load_all_ratings(all_users=True)
        if not all_data.empty:
            all_data = all_data[all_data['user_id'] != st.session_state.user_id]
            
        combined_data = pd.concat([all_data, new_rating], ignore_index=True)
        combined_data = combined_data.sort_values('timestamp', ascending=False)
        combined_data = combined_data.drop_duplicates(subset=['user_id', 'dish_id'], keep='first')
        
        ratings_file = pathlib.Path(__file__).parent / APP_CONFIG["ratings_file"]
        combined_data.to_excel(ratings_file, index=False)
        
        # 更新会话状态中的用户评分
        user_ratings = combined_data[combined_data['user_id'] == st.session_state.user_id].copy()
        st.session_state.user_ratings = user_ratings.to_dict('records')
        
        return True, "评分保存成功"
    
    except Exception as e:
        return False, f"评分保存失败: {str(e)}"

def load_collaborative_filtering_model():
    """加载协同过滤模型（带缓存）"""
    try:
        ratings_file = pathlib.Path(__file__).parent / APP_CONFIG["ratings_file"]
        if ratings_file.exists():
            data_df = load_all_ratings(all_users=True)
            
            if len(data_df) < 10:
                return None
                
            reader = Reader(line_format='user item rating', rating_scale=(1, 5))
            data = Dataset.load_from_df(data_df[['user_id', 'dish_id', 'rating']], reader)
            trainset = data.build_full_trainset()
            
            algo = SVD(random_state=42, n_factors=100, n_epochs=5)
            algo.fit(trainset)
            return algo
        return None
    except Exception as e:
        st.warning(f"协同过滤模型加载失败: {e}")
        return None

# --------------------- 辅助函数 ---------------------
def predict_dish(image, model):
    """使用模型预测菜品"""
    if model is None:
        return "模型加载失败", 0, []
        
    try:
        img = PILImage.create(image)
        pred, pred_idx, probs = model.predict(img)
        
        if pred not in dish_names:
            pred = dish_names[np.argmax(probs)]
            
        return pred, probs[pred_idx].item(), probs
    except Exception as e:
        st.error(f"菜品识别出错: {e}")
        return "识别失败", 0, []

def display_dish_info(dish_name):
    """获取菜品详细信息"""
    if dish_name not in dish_id_map:
        return {
            "名称": dish_name,
            "菜系": "未知",
            "口味": "未知",
            "卡路里": "未知",
            "描述": "暂无详细信息",
            "推荐人群": "未知",
            "禁忌人群": "未知",
            "image": None
        }
        
    dish_info = dishes_df[dishes_df['dish_name'] == dish_name].iloc[0]
    return {
        "名称": dish_name,
        "菜系": dish_info['cuisine'],
        "口味": dish_info['taste'],
        "卡路里": f"{dish_info['calorie']}大卡每100克",
        "描述": dish_info['description'],
        "推荐人群": dish_info['recommended population'],
        "禁忌人群": dish_info['contraindicated population'],
        "image": dish_info.get('image', None)
    }

def set_page_style():
    """设置页面样式"""
    st.markdown("""
    <style>
    .centered-title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #FF6B6B;
        margin: 20px 0;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .rating-stars {
        color: #FFD700;
        font-size: 24px;
    }
    .recommendation-card {
        border-left: 4px solid #FF6B6B;
        padding-left: 15px;
        margin-bottom: 15px;
    }
    .highlight {
        color: #FF6B6B;
        font-weight: bold;
    }
    .error-message {
        color: red;
        font-weight: bold;
    }
    .sidebar-title {
        font-size: 20px;
        font-weight: bold;
        color: #FF6B6B;
        margin: 10px 0;
    }
    .user-id-badge {
        display: inline-block;
        background-color: #f0f0f0;
        padding: 5px 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 200px;
    }
    </style>
    """, unsafe_allow_html=True)

def get_download_link(df, filename):
    """生成下载链接"""
    try:
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">下载评分数据</a>'
        return href
    except:
        return "下载失败，请重试"

# --------------------- 页面函数 ---------------------
def home_page():
    """首页"""
    st.markdown('<div class="centered-title">🍱 食堂菜品识别系统</div>', unsafe_allow_html=True)
    
    # 显示用户ID和专属链接
    st.markdown(f"""
    <div class="user-id-badge">
        <b>您的用户ID:</b> {st.session_state.user_id}
    </div>
    """, unsafe_allow_html=True)
    
    # 获取当前URL信息
    current_params = st.experimental_get_query_params()
    
    # 获取基础URL（不包含查询参数）
    base_url = st.get_option('server.baseUrlPath')
    if base_url.endswith('/'):
        base_url = base_url[:-1]
    
    # 生成带user_id的专属链接
    专属链接 = f"{base_url}?user_id={st.session_state.user_id}"
    
    st.markdown(f"""
    <a href="{专属链接}">
        <button style="margin-top:10px;">复制您的专属链接</button>
    </a>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    这是一个基于协同过滤算法的食堂菜品识别与推荐系统。您可以上传菜品图片，系统将识别菜品并为您提供菜品详细信息，在您食用过后可以对菜品进行评分，
    评分后系统会根据您的口味偏好为您推荐其他菜品,祝您用餐愉快🍽️🍽️🍽️!
    """, unsafe_allow_html=True)
    
    # 显示系统状态
    with st.expander("系统状态"):
        st.write(f"当前用户: {st.session_state.user_id}")
        st.write(f"已评价菜品: {len(st.session_state.user_ratings)}")
        st.write(f"系统时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

def dish_recognition_page():
    """菜品识别页面"""
    st.markdown('<div class="centered-title">🍽️ 菜品识别</div>', unsafe_allow_html=True)
    
    if model is None:
        st.error("菜品识别模型加载失败，请检查文件路径或联系管理员")
        return
    
    st.subheader("上传菜品图片")
    uploaded_file = st.file_uploader("选择图片", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(uploaded_file, caption="上传的菜品图片", use_container_width=True)
        
        with st.spinner("正在识别菜品..."):
            try:
                img = PILImage.create(uploaded_file)
                if img.size[0] < 50 or img.size[1] < 50:
                    st.warning("图片尺寸过小，可能影响识别准确率")
                    
                pred_dish, confidence, probs = predict_dish(img)
                st.markdown(f"识别结果: <span class='highlight'>{pred_dish}</span> (置信度: {confidence*100:.2f}%)", unsafe_allow_html=True)
                
                st.subheader("菜品介绍")
                dish_info = display_dish_info(pred_dish)
                for key, value in dish_info.items():
                    if key != "image":
                        st.markdown(f"**{key}:** {value}")
                
                st.subheader("识别概率分布")
                valid_dishes = [dish for dish in dish_names if dish in dishes_df['dish_name'].values]
                filtered_probs = [probs[i] for i, dish in enumerate(dish_names) if dish in valid_dishes]
                
                top5 = sorted(zip(valid_dishes, filtered_probs), key=lambda x: x[1], reverse=True)[:5]
                labels = [item[0] for item in top5]
                values = [item[1] for item in top5]
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(labels, values, color='tomato')
                ax.set_ylabel('概率')
                ax.set_title('菜品识别概率分布')
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)
                
                # 评分功能
                st.subheader("评价该菜品")
                rating = st.slider("请给出评分 (1-5星)", 1, 5, 3)
                
                if st.button("提交评分"):
                    dish_id = dish_id_map.get(pred_dish, 0)
                    if dish_id == 0:
                        st.error(f"未找到菜品 {pred_dish} 的ID映射,评分失败")
                        return
                        
                    success, message = save_rating_safely(
                        dish_id=dish_id,
                        rating=rating
                    )
                    
                    if success:
                        st.success(f"感谢评分！您给{pred_dish}打了{rating}星")
                        st.markdown(f"<div class='rating-stars'>{'⭐' * rating}</div>", unsafe_allow_html=True)
                        
                        # 重新加载协同过滤模型
                        st.session_state['collab_model'] = load_collaborative_filtering_model()
                        
                        # 提供下载链接
                        if st.session_state.user_ratings:
                            ratings_df = pd.DataFrame(st.session_state.user_ratings)
                            st.markdown(get_download_link(ratings_df, f'user_{st.session_state.user_id}_ratings.csv'), unsafe_allow_html=True)
                    else:
                        st.error(message)

            except Exception as e:
                st.error(f"图片处理出错: {e}")

def recommendation_page():
    """推荐页面"""
    st.markdown('<div class="centered-title">📋 菜品推荐</div>', unsafe_allow_html=True)
    
    if not st.session_state.user_ratings or len(st.session_state.user_ratings) == 0:
        st.warning("您还没有评分记录，请先识别并评价菜品，以便获取个性化推荐")
        return
    
    st.subheader("为您推荐菜品")
    with st.spinner("正在生成推荐..."):
        try:
            current_algo = st.session_state.get('collab_model', load_collaborative_filtering_model())
            
            if not current_algo:
                st.info("评分数据不足，使用基础推荐")
                rated_dish_ids = [r['dish_id'] for r in st.session_state.user_ratings]
                recommended_dishes = dishes_df[~dishes_df['dish_id'].isin(rated_dish_ids)].sample(3)
                
                st.success("为您推荐（基础推荐）：")
                for i, row in recommended_dishes.iterrows():
                    with st.container():
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <h4>**{i+1}. {row['dish_name']}** ({row['cuisine']})</h4>
                            <p>口味：{row['taste']} | 卡路里：{row['calorie']}大卡</p>
                            <p>描述：{row['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                new_user_ratings = pd.DataFrame(st.session_state.user_ratings)
                all_dish_ids = dishes_df['dish_id'].tolist()
                rated_dish_ids = new_user_ratings['dish_id'].tolist()
                unrated_dish_ids = [d for d in all_dish_ids if d not in rated_dish_ids]
                
                predictions = []
                for dish_id in unrated_dish_ids:
                    if dish_id in dishes_df['dish_id'].values:
                        pred = current_algo.predict(uid=st.session_state.user_id, iid=dish_id)
                        predictions.append((dish_id, pred.est))
                
                if predictions:
                    predictions_df = pd.DataFrame(predictions, columns=['dish_id', 'predicted_rating'])
                    recommendations = pd.merge(
                        predictions_df,
                        dishes_df[['dish_id', 'dish_name', 'cuisine', 'taste', 'calorie', 'description']],
                        on='dish_id'
                    ).sort_values('predicted_rating', ascending=False)
                    
                    st.success("为您推荐（协同过滤）：")
                    for i, row in recommendations.head(3).iterrows():
                        with st.container():
                            st.markdown(f"""
                            <div class="recommendation-card">
                                <h4>**{i+1}. {row['dish_name']}** ({row['cuisine']})</h4>
                                <p>预测评分：{row['predicted_rating']:.2f}星 | 口味：{row['taste']} | 卡路里：{row['calorie']}大卡</p>
                                <p>描述：{row['description']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("没有可推荐的菜品，请尝试评价更多菜品")
                    
        except Exception as e:
            st.error(f"推荐生成失败: {e}")

def rating_statistics_page():
    """评分统计页面"""
    st.markdown('<div class="centered-title">📊 评分统计</div>', unsafe_allow_html=True)
    
    if not st.session_state.user_ratings or len(st.session_state.user_ratings) == 0:
        st.warning("您还没有评分记录")
        return
    
    ratings_df = pd.DataFrame(st.session_state.user_ratings)
    
    st.subheader("评分分布")
    rating_counts = ratings_df['rating'].value_counts().sort_index()
    st.bar_chart(rating_counts)
    
    st.subheader("您最喜欢的菜品")
    if 'dish_id' in ratings_df.columns and 'dish_name' in dishes_df.columns:
        most_liked = ratings_df.groupby('dish_id')['rating'].mean().nlargest(3)
        for dish_id, score in most_liked.items():
            dish_name = dishes_df[dishes_df['dish_id'] == dish_id]['dish_name'].iloc[0]
            st.markdown(f"- {dish_name}: {score:.2f}星")

def test_page():
    st.markdown('<div class="centered-title">🧪 系统测试</div>', unsafe_allow_html=True)
    
    # 让用户上传测试图片
    test_img = st.file_uploader("上传测试菜品图片（如地三鲜）", type=["jpg", "png", "jpeg"])
    
    if test_img and st.button("运行测试"):
        try:
            img = PILImage.create(test_img)
            pred, conf, _ = predict_dish(img)
            st.write(f"测试预测结果: {pred} (置信度: {conf*100:.2f}%)")
        except Exception as e:
            st.error(f"测试失败: {e}")
    elif test_img is None and st.button("运行测试"):
        st.error("请先上传测试图片")

# --------------------- 主程序 ---------------------
def main():
    set_page_style()
    
    # 侧边栏导航（添加小表情）
    st.sidebar.markdown('<div class="sidebar-title">🍱 导航菜单</div>', unsafe_allow_html=True)
    page_options = [
        "🏠 首页", 
        "🍽️ 菜品识别", 
        "📋 菜品推荐", 
        "📊 评分统计", 
        "🧪 系统测试"
    ]
    selected_page = st.sidebar.radio("选择页面", page_options)
    
    # 去除表情获取原始页面名称
    page_mapping = {
        "🏠 首页": "首页",
        "🍽️ 菜品识别": "菜品识别",
        "📋 菜品推荐": "菜品推荐",
        "📊 评分统计": "评分统计",
        "🧪 系统测试": "系统测试"
    }
    original_page = page_mapping.get(selected_page, selected_page)
    st.session_state.current_page = original_page
    
    # 显示对应页面
    if original_page == "首页":
        home_page()
    elif original_page == "菜品识别":
        dish_recognition_page()
    elif original_page == "菜品推荐":
        recommendation_page()
    elif original_page == "评分统计":
        rating_statistics_page()
    elif original_page == "系统测试":
        test_page()
    
    # 页脚
    st.markdown("---")
    st.write("食堂菜品识别系统 🍽️ | yeah!")

# 加载模型和数据
model = load_model()
try:
    dishes_file = pathlib.Path(__file__).parent / APP_CONFIG["dishes_file"]
    dishes_df = pd.read_excel(dishes_file)
    dish_names = model.dls.vocab
    dish_id_map = {}
    for idx, row in dishes_df.iterrows():
        dish_name = row['dish_name']
        if dish_name in dish_names:
            dish_id_map[dish_name] = row.get('dish_id', idx + 1)
    
    missing_dishes = [d for d in dish_names if d not in dish_id_map]
    if missing_dishes:
        st.warning(f"警告: 菜品信息表中缺少以下模型类别: {', '.join(missing_dishes)}")
        
except Exception as e:
    st.error(f"菜品信息加载失败: {e}")
    st.stop()

if __name__ == "__main__":
    main()