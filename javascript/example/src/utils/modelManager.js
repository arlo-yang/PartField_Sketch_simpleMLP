import { initialModels } from '../models-config.js';
import { setColor, updateRenderMode } from './visualManager.js';
import { removeArrowObject } from './arrow.js';
import { addScreenshotControl } from './seg_screenshot.js';
import { uncheckAllRenders } from './uiControls.js';
import { viewPositions } from './viewerSetup.js';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

// 模型信息管理
class ModelManager {
    constructor() {
        this.categories = new Map();
        this.currentModelInfo = null;
        this.sliders = {};
    }

    // 初始化URDF事件监听
    initURDFEvents(viewer) {
        console.group('Initializing URDF Events');
        
        // 当 URDF 模型变化时，清除所有滑块
        viewer.addEventListener('urdf-change', () => {
            Object.values(this.sliders).forEach(sl => sl.remove());
            this.sliders = {};
        });

        // 当关节角度发生变化时，更新对应滑块显示
        viewer.addEventListener('angle-change', e => {
            if (this.sliders[e.detail]) this.sliders[e.detail].update();
        });

        // 鼠标悬停在关节上时，高亮对应列表项
        viewer.addEventListener('joint-mouseover', e => {
            const j = document.querySelector(`li[joint-name="${e.detail}"]`);
            if (j) j.setAttribute('robot-hovered', true);
        });

        // 鼠标离开时，移除高亮
        viewer.addEventListener('joint-mouseout', e => {
            const j = document.querySelector(`li[joint-name="${e.detail}"]`);
            if (j) j.removeAttribute('robot-hovered');
        });

        // 操作关节开始时，滚动到对应列表项，并禁用自动居中
        viewer.addEventListener('manipulate-start', e => {
            const j = document.querySelector(`li[joint-name="${e.detail}"]`);
            if (j) {
                j.scrollIntoView({ block: 'nearest' });
                window.scrollTo(0, 0);
            }
            viewer.noAutoRecenter = true;
        });

        // 关节操作结束时，恢复自动居中
        viewer.addEventListener('manipulate-end', () => {
            viewer.noAutoRecenter = false;
        });

        // URDF 加载完毕后的处理
        viewer.addEventListener('urdf-processed', () => {
            this.handleURDFProcessed(viewer);
        });

        console.log('URDF Events initialized');
        console.groupEnd();
    }

    // 处理URDF加载完成后的设置
    handleURDFProcessed(viewer) {
        const robot = viewer.robot;
        if (!robot) return;

        const hasPrismaticJoint = Object.values(robot.joints)
            .some(joint => joint.jointType === 'prismatic');
        const hasRevoluteJoint = Object.values(robot.joints)
            .some(joint => joint.jointType === 'revolute');

        if (hasPrismaticJoint || hasRevoluteJoint) {
            // 若存在滑动或旋转关节，使用 top-left 视角
            const initialView = window.viewPositions ? window.viewPositions['top-left'] : {
                position: new THREE.Vector3(),
                target: new THREE.Vector3()
            };
            viewer.camera.position.copy(initialView.position);
            viewer.camera.lookAt(initialView.target);

            if (viewer.plane) {
                viewer.plane.position.y = -0.5;
            }

            // 设置关节初始值
            Object.values(robot.joints).forEach(joint => {
                if (joint.jointType === 'prismatic' || joint.jointType === 'revolute') {
                    joint.setJointValue(0);
                }
            });

            robot.rotation.y = Math.PI;
            robot.updateMatrixWorld(true);
        }
    }

    // 获取当前模型完整信息
    getCurrentModelInfo() {
        if (!window.currentUrdfPath) {
            console.log('No current URDF path');
            return null;
        }

        const currentModel = document.querySelector(`.model-item[urdf="${window.currentUrdfPath}"]`);
        if (!currentModel) {
            console.log('No model element found');
            return null;
        }

        const categoryContainer = currentModel.closest('.category');
        if (!categoryContainer) {
            console.log('No category container found');
            return null;
        }

        const headerElement = categoryContainer.querySelector('.category-header');
        const category = headerElement
            ?.textContent
            ?.split('(')[0]
            ?.replace(/[▶▼]/g, '')
            ?.trim()
            ?.toLowerCase();

        return {
            id: currentModel.textContent.trim(),
            type: category || this.getModelTypeFromPath(window.currentUrdfPath),
            urdfPath: window.currentUrdfPath,
            element: currentModel
        };
    }

    // 从路径获取模型类型
    getModelTypeFromPath(urdfPath) {
        if (!urdfPath) return 'unknown';
        const pathParts = urdfPath.split('/');
        return pathParts[pathParts.length - 3]?.toLowerCase() || 'unknown';
    }

    // 初始化模型分类
    async initializeCategories(viewer) {
        console.group('Initializing Model Categories');
        const categoriesContainer = document.getElementById('urdf-categories');
        
        // 清空现有内容
        categoriesContainer.innerHTML = '';

        // 收集所有模型信息
        for (const model of initialModels) {
            try {
                let category = 'Unknown';
                let modelId = '';
                const response = await fetch(model.meta);
                const metaData = await response.json();
                category = metaData.model_cat;
                modelId = model.urdf.split('/').slice(-2)[0];

                if (!this.categories.has(category)) {
                    this.categories.set(category, []);
                }

                this.categories.get(category).push({
                    id: modelId,
                    numericId: parseInt(modelId),
                    urdf: model.urdf,
                    color: model.color
                });
            } catch (error) {
                console.error('Error loading model info:', error);
            }
        }

        // 创建分类界面
        for (const [category, models] of this.categories) {
            const categoryDiv = this.createCategoryElement(category, models, viewer);
            categoriesContainer.appendChild(categoryDiv);
        }

        // 标记当前选中的模型（如果有）
        if (window.currentUrdfPath) {
            const currentModelElements = document.querySelectorAll(`.model-item[urdf="${window.currentUrdfPath}"]`);
            currentModelElements.forEach(element => {
                element.classList.add('selected');
            });
            
            // 展开包含当前选中模型的分类
            const categoryContent = currentModelElements[0]?.closest('.category-content');
            if (categoryContent) {
                categoryContent.classList.add('expanded');
                const arrow = categoryContent.previousElementSibling.querySelector('.arrow');
                if (arrow) {
                    arrow.textContent = '▼';
                }
            }
        }

        console.log('Categories initialized successfully');
        console.groupEnd();
    }

    // 创建分类元素
    createCategoryElement(category, models, viewer) {
        const categoryDiv = document.createElement('div');
        categoryDiv.className = 'category';

        // 按数字ID排序
        models.sort((a, b) => a.numericId - b.numericId);

        const headerDiv = document.createElement('div');
        headerDiv.className = 'category-header';
        headerDiv.innerHTML = `<span class="arrow">▶</span>${category} (${models.length})`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'category-content';

        // 为每个模型创建项
        models.forEach(model => {
            const modelDiv = this.createModelElement(model, viewer);
            contentDiv.appendChild(modelDiv);
        });

        // 折叠/展开功能
        headerDiv.addEventListener('click', () => {
            contentDiv.classList.toggle('expanded');
            headerDiv.querySelector('.arrow').textContent =
                contentDiv.classList.contains('expanded') ? '▼' : '▶';
        });

        categoryDiv.appendChild(headerDiv);
        categoryDiv.appendChild(contentDiv);
        return categoryDiv;
    }

    // 创建模型元素
    createModelElement(model, viewer) {
        const modelDiv = document.createElement('div');
        modelDiv.className = 'model-item';
        modelDiv.textContent = `${model.id}`;
        modelDiv.setAttribute('urdf', model.urdf);
        modelDiv.setAttribute('color', model.color);

        modelDiv.addEventListener('click', () => this.handleModelClick(model, viewer));
        return modelDiv;
    }

    // 处理模型点击事件
    handleModelClick(model, viewer) {
        console.group('Model Click Handler');
        try {
            // 记录当前状态
            console.log('Current state:', {
                currentRenderMode: window.currentRenderMode,
                hasRobot: !!viewer.robot,
                newModelPath: model.urdf
            });

            // 清除之前选中模型的标记
            const previousSelected = document.querySelector('.model-item.selected');
            if (previousSelected) {
                previousSelected.classList.remove('selected');
            }

            // 标记当前选中的模型
            const allModels = document.querySelectorAll(`.model-item[urdf="${model.urdf}"]`);
            allModels.forEach(modelElement => {
                modelElement.classList.add('selected');
            });

            // 确保所有mesh可见性恢复
            if (viewer.robot) {
                viewer.robot.traverse(child => {
                    if (child.isMesh) {
                        child.visible = true;
                    }
                });
            }

            // 若当前为 Arrow 模式，先切回 default 模式
            if (window.currentRenderMode && window.currentRenderMode.startsWith('arrow')) {
                console.log('Cleaning up arrow mode...');
                removeArrowObject(viewer);
                uncheckAllRenders();
                const defaultRender = document.getElementById('default-render');
                defaultRender.classList.add('checked');
                window.currentRenderMode = 'default';
            }


            const currentRenderModeBtn = document.querySelector('.toggle.checked')?.id || 'default-render';
            const currentState = document.getElementById(
                currentRenderModeBtn.replace('-render', '-state')
            )?.value || 'original';

            console.log('Switching to new model with:', {
                renderMode: currentRenderModeBtn,
                state: currentState
            });

            window.currentUrdfPath = model.urdf;

            // 确保正确设置包路径
            if (!/^https?:\/\//.test(model.urdf)) {
                viewer.package = model.urdf.split('/').slice(0, -2).join('/');
            }

            viewer.up = '-Z';
            viewer.urdf = model.urdf;
            setColor(model.color);

            viewer.camera.position.set(2, 2, 2);
            viewer.camera.lookAt(0, 0, 0);

            // 保存模型ID，用于记录归一化状态
            window.currentModelId = model.id;
            
            // 检查该模型是否已经被归一化过
            const normalizedModels = JSON.parse(localStorage.getItem('normalizedModels') || '{}');
            const wasNormalized = normalizedModels[model.id] === true;
            console.log(`模型 ${model.id} 的归一化状态: ${wasNormalized ? '已归一化' : '未归一化'}`);

            viewer.addEventListener('urdf-processed', () => {
                console.log('New URDF processed, updating render mode...');
                updateRenderMode(
                    `${currentRenderModeBtn.replace('-render', '')}-${currentState}`,
                    viewer,
                    window.currentUrdfPath,
                    window.currentRenderMode,
                    window.defaultLight
                );
                
                // 触发自定义事件，通知几何信息模块更新
                if (currentRenderModeBtn === 'default-render') {
                    const event = new CustomEvent('model-geometry-update');
                    viewer.dispatchEvent(event);
                }
            }, { once: true });

            addScreenshotControl(viewer, window.currentUrdfPath);

        } catch (error) {
            console.error('Error in model click handler:', error);
        }
        console.groupEnd();
    }
}

// 创建单例实例
const modelManager = new ModelManager();

// 导出函数和实例
export const getCurrentModelInfo = () => modelManager.getCurrentModelInfo();
export const initializeCategories = (viewer) => modelManager.initializeCategories(viewer);
export const getModelTypeFromPath = (path) => modelManager.getModelTypeFromPath(path);
export const initURDFEvents = (viewer) => modelManager.initURDFEvents(viewer); 