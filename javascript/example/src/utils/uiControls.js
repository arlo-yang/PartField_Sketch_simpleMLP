import * as THREE from 'three';
import { updateRenderMode } from './visualManager.js';
import { removeArrowObject, loadArrowModel, applyArrowMaterial, handleArrowJointIdChange, getAvailableJointIds } from './arrow.js';
import { applyURDFMaterialAndState, depthMaterial } from './visualManager.js';
import { addScreenshotControl, addGalleryControl } from './seg_screenshot.js';
import { addAxesControl, addViewControl } from './viewerSetup.js';
import { registerDragEvents } from '../dragAndDrop.js';
import { clearHighlights } from './urdfInfo.js';

// 全局变量，用于存储当前的渲染模式
window.currentRenderMode = 'default';

/**
 * 触发渲染模式变化事件
 * 用于通知其他组件当前渲染模式已经改变
 */
function triggerRenderModeChanged() {
    const event = new CustomEvent('renderModeChanged', {
        detail: { mode: window.currentRenderMode }
    });
    window.dispatchEvent(event);
}

/**
 * 更新分段模式的关节下拉菜单
 * @param {Object} viewer - URDF查看器对象
 */
function updateJointDropdown(viewer) {
    const jointSelect = document.getElementById('segmentation-joint-select');
    if (!jointSelect) {
        return;
    }
    
    // 清空现有选项，只保留"unselect"
    while (jointSelect.options.length > 1) {
        jointSelect.remove(1);
    }
    
    // 如果没有viewer或robot，直接返回
    if (!viewer || !viewer.robot || !viewer.robot.joints) {
        return;
    }
    
    const joints = viewer.robot.joints;
    const jointCount = Object.keys(joints).length;
    
    // 添加关节选项
    let index = 0;
    for (const [jointName, joint] of Object.entries(joints)) {
        // 只添加非fixed类型的关节
        if (joint.jointType !== 'fixed') {
            const option = document.createElement('option');
            option.value = jointName;
            option.textContent = jointName;
            jointSelect.appendChild(option);
            index++;
        }
    }
    
    return index;
}

export function uncheckAllRenders() {
    const axesVisible = window.axesController && window.axesController.isVisible();
    const viewVisible = window.viewController && window.viewController.isVisible();

    const renderButtons = [
        'default-render',
        'normal-render',
        'depth-render',
        'wireframe-render',
        'segmentation-render',
        'arrow-default',
        'arrow-sketch',
        'arrow-wireframe'
    ];

    renderButtons.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.classList.remove('checked');
        }
    });

    if (axesVisible) {
        window.axesController.show();
    }
    if (viewVisible) {
        window.viewController.show();
    }
}

// 更新箭头样式选择器
export function updateArrowStyleSelectors(urdfPath) {
    const jointIds = getAvailableJointIds(urdfPath);
    
    // Get all arrow style selectors
    const selectors = [
        document.getElementById('arrow-default-style'),
        document.getElementById('arrow-sketch-style'),
        document.getElementById('arrow-wireframe-style')
    ];
    
    // Update each selector with available joint IDs
    selectors.forEach(selector => {
        if (!selector) return;
        
        // Save current selection if possible
        const currentValue = selector.value;
        
        // Clear existing options
        selector.innerHTML = '';
        
        // Add new options
        jointIds.forEach(jointId => {
            const option = document.createElement('option');
            option.value = jointId;
            option.textContent = `joint_${jointId}`; // 改为joint_id格式
            selector.appendChild(option);
        });
        
        // Restore previous selection if it exists in new options
        if (jointIds.includes(currentValue)) {
            selector.value = currentValue;
        }
    });
}

export function initUIControls(viewer, defaultLight) {
    try {
        // 获取所有控制按钮
        const defaultRender = document.getElementById('default-render');
        const normalRender = document.getElementById('normal-render');
        const depthRender = document.getElementById('depth-render');
        const wireframeRender = document.getElementById('wireframe-render');
        const segmentationRender = document.getElementById('segmentation-render');
        const arrowDefault = document.getElementById('arrow-default');
        const arrowSketch = document.getElementById('arrow-sketch');
        const arrowWireframe = document.getElementById('arrow-wireframe');

        // 获取所有状态选择器
        const defaultState = document.getElementById('default-state');
        const normalState = document.getElementById('normal-state');
        const depthState = document.getElementById('depth-state');
        const wireframeState = document.getElementById('wireframe-state');
        const segmentationJointSelect = document.getElementById('segmentation-joint-select');
        
        // 获取箭头样式选择器
        const arrowDefaultStyle = document.getElementById('arrow-default-style');

        // 检查元素是否存在
        if (!defaultRender || !normalRender || !depthRender || !wireframeRender || 
            !segmentationRender || !arrowDefault) {
            return;
        }

        // 渲染模式切换事件监听
        defaultRender.addEventListener('click', () => {
            uncheckAllRenders();
            defaultRender.classList.add('checked');
            window.currentRenderMode = 'default';
            removeArrowObject(viewer);
            
            // 确保URDF模型可见
            if (viewer.robot) {
                viewer.robot.visible = true;
            }
            
            updateRenderMode('default', viewer, window.currentUrdfPath, window.currentRenderMode, defaultLight);
            triggerRenderModeChanged(); // 触发渲染模式变化事件
        });

        defaultState.addEventListener('change', () => {
            if (defaultRender.classList.contains('checked')) {
                updateRenderMode('default-' + defaultState.value, viewer, window.currentUrdfPath, window.currentRenderMode, defaultLight);
            }
        });

        normalRender.addEventListener('click', () => {
            try {
                uncheckAllRenders();
                normalRender.classList.add('checked');
                window.currentRenderMode = 'normal';
                removeArrowObject(viewer);
                
                // 确保URDF模型可见
                if (viewer.robot) {
                    viewer.robot.visible = true;
                }
                
                updateRenderMode('normal', viewer, window.currentUrdfPath, window.currentRenderMode, defaultLight);
                triggerRenderModeChanged(); // 触发渲染模式变化事件
            } catch (error) {
                // 错误处理
            }
        });

        normalState.addEventListener('change', () => {
            if (normalRender.classList.contains('checked')) {
                updateRenderMode('normal-' + normalState.value, viewer, window.currentUrdfPath, window.currentRenderMode, defaultLight);
            }
        });

        depthRender.addEventListener('click', () => {
            try {
                uncheckAllRenders();
                depthRender.classList.add('checked');
                window.currentRenderMode = 'depth';
                removeArrowObject(viewer);
                
                // 确保URDF模型可见
                if (viewer.robot) {
                    viewer.robot.visible = true;
                }
                
                updateRenderMode('depth-' + depthState.value, viewer, window.currentUrdfPath, window.currentRenderMode, defaultLight);
                triggerRenderModeChanged(); // 触发渲染模式变化事件
            } catch (error) {
                // 错误处理
            }
        });

        wireframeRender.addEventListener('click', () => {
            try {
                uncheckAllRenders();
                wireframeRender.classList.add('checked');
                window.currentRenderMode = 'wireframe';
                removeArrowObject(viewer);
                
                // 确保URDF模型可见
                if (viewer.robot) {
                    viewer.robot.visible = true;
                }
                
                updateRenderMode('wireframe-' + wireframeState.value, viewer, window.currentUrdfPath, window.currentRenderMode, defaultLight);
                triggerRenderModeChanged(); // 触发渲染模式变化事件
            } catch (error) {
                // 错误处理
            }
        });

        segmentationRender.addEventListener('click', () => {
            try {
                uncheckAllRenders();
                segmentationRender.classList.add('checked');
                window.currentRenderMode = 'segmentation';
                removeArrowObject(viewer);
                
                // 确保URDF模型可见
                if (viewer.robot) {
                    viewer.robot.visible = true;
                }
                
                // 更新关节下拉菜单
                updateJointDropdown(viewer);
                
                // 使用当前选中的关节
                const selectedJoint = segmentationJointSelect.value;
                
                // 更简洁的方式处理unselect
                if (selectedJoint === 'unselect') {
                    // 未选择任何关节
                    window.currentSelectedJoint = null;
                    updateRenderMode('segmentation', viewer, window.currentUrdfPath, window.currentRenderMode, defaultLight);
                } else {
                    // 选择了特定关节
                    window.currentSelectedJoint = selectedJoint;
                    updateRenderMode(`segmentation-${selectedJoint}`, viewer, window.currentUrdfPath, window.currentRenderMode, defaultLight);
                }
                
                triggerRenderModeChanged(); // 触发渲染模式变化事件
            } catch (error) {
                // 错误处理
            }
        });

        segmentationJointSelect.addEventListener('change', () => {
            if (segmentationRender.classList.contains('checked')) {
                const selectedJoint = segmentationJointSelect.value;
                
                // 更简洁的方式处理unselect
                if (selectedJoint === 'unselect') {
                    // 未选择任何关节
                    window.currentSelectedJoint = null;
                    updateRenderMode('segmentation', viewer, window.currentUrdfPath, window.currentRenderMode, defaultLight);
                } else {
                    // 选择了特定关节
                    window.currentSelectedJoint = selectedJoint;
                    updateRenderMode(`segmentation-${selectedJoint}`, viewer, window.currentUrdfPath, window.currentRenderMode, defaultLight);
                }
            }
        });

        // 监听URDF处理完成事件，更新关节下拉菜单
        viewer.addEventListener('urdf-processed', () => {
            updateJointDropdown(viewer);
        });

        // 初始化箭头样式选择器
        if (window.currentUrdfPath) {
            updateArrowStyleSelectors(window.currentUrdfPath);
        }
        
        // 监听模型变化，更新箭头样式选择器
        viewer.addEventListener('urdf-processed', () => {
            if (window.currentUrdfPath) {
                updateArrowStyleSelectors(window.currentUrdfPath);
            }
        });

        arrowDefault.addEventListener('click', () => {
            uncheckAllRenders();
            arrowDefault.classList.add('checked');
            window.currentRenderMode = 'arrow-default';
            window.currentJointId = arrowDefaultStyle.value;
            
            // 清除任何现有的高亮
            if (typeof clearHighlights === 'function') {
                clearHighlights();
            }
            
            // 隐藏URDF模型
            if (viewer.robot) {
                viewer.robot.visible = false;
            }
            
            updateRenderMode(`arrow-default-${arrowDefaultStyle.value}`, viewer, window.currentUrdfPath, window.currentRenderMode, defaultLight);
            triggerRenderModeChanged(); // 触发渲染模式变化事件
        });

        // 添加arrow-sketch按钮事件处理
        if (arrowSketch) {
            arrowSketch.addEventListener('click', () => {
                uncheckAllRenders();
                arrowSketch.classList.add('checked');
                window.currentRenderMode = 'arrow-sketch';
                window.currentJointId = arrowDefaultStyle.value;
                
                // 清除任何现有的高亮
                if (typeof clearHighlights === 'function') {
                    clearHighlights();
                }
                
                loadArrowModel(window.currentUrdfPath, viewer, () => {
                    // 隐藏URDF模型
                    if (viewer.robot) {
                        viewer.robot.visible = false;
                    }
                    
                    applyArrowMaterial('arrow-sketch');
                    viewer.redraw();
                });
                triggerRenderModeChanged(); // 触发渲染模式变化事件
            });
        }
        
        // 添加arrow-wireframe按钮事件处理
        if (arrowWireframe) {
            arrowWireframe.addEventListener('click', () => {
                uncheckAllRenders();
                arrowWireframe.classList.add('checked');
                window.currentRenderMode = 'arrow-wireframe';
                window.currentJointId = arrowDefaultStyle.value;
                
                // 清除任何现有的高亮
                if (typeof clearHighlights === 'function') {
                    clearHighlights();
                }
                
                loadArrowModel(window.currentUrdfPath, viewer, () => {
                    // 隐藏URDF模型
                    if (viewer.robot) {
                        viewer.robot.visible = false;
                    }
                    
                    applyArrowMaterial('arrow-wireframe');
                    viewer.redraw();
                });
                triggerRenderModeChanged(); // 触发渲染模式变化事件
            });
        }
        
        // 添加箭头样式选择器的事件监听
        arrowDefaultStyle.addEventListener('change', () => {
            if (arrowDefault.classList.contains('checked')) {
                window.currentJointId = arrowDefaultStyle.value;
                // 清除任何现有的高亮
                clearHighlights();
                handleArrowJointIdChange(viewer, window.currentUrdfPath, window.currentRenderMode, 'arrow-default', arrowDefaultStyle.value);
            }
        });
        
        // 添加arrow-sketch样式选择器的事件监听
        const arrowSketchStyle = document.getElementById('arrow-sketch-style');
        if (arrowSketchStyle) {
            arrowSketchStyle.addEventListener('change', () => {
                if (arrowSketch && arrowSketch.classList.contains('checked')) {
                    window.currentJointId = arrowSketchStyle.value;
                    // 清除任何现有的高亮
                    if (typeof clearHighlights === 'function') {
                        clearHighlights();
                    }
                    handleArrowJointIdChange(viewer, window.currentUrdfPath, window.currentRenderMode, 'arrow-sketch', arrowSketchStyle.value);
                }
            });
        }
        
        // 添加arrow-wireframe样式选择器的事件监听
        const arrowWireframeStyle = document.getElementById('arrow-wireframe-style');
        if (arrowWireframeStyle) {
            arrowWireframeStyle.addEventListener('change', () => {
                if (arrowWireframe && arrowWireframe.classList.contains('checked')) {
                    window.currentJointId = arrowWireframeStyle.value;
                    // 清除任何现有的高亮
                    if (typeof clearHighlights === 'function') {
                        clearHighlights();
                    }
                    handleArrowJointIdChange(viewer, window.currentUrdfPath, window.currentRenderMode, 'arrow-wireframe', arrowWireframeStyle.value);
                }
            });
        }

        depthState.addEventListener('change', () => {
            if (depthRender.classList.contains('checked')) {
                // 确保状态值被正确传递
                updateRenderMode(
                    `depth-${depthState.value}`, 
                    viewer, 
                    window.currentUrdfPath, 
                    window.currentRenderMode, 
                    defaultLight
                );
                viewer.redraw(); // 强制重新渲染
            }
        });

        wireframeState.addEventListener('change', () => {
            if (wireframeRender.classList.contains('checked')) {
                updateRenderMode('wireframe-' + wireframeState.value, viewer, window.currentUrdfPath, window.currentRenderMode, defaultLight);
            }
        });

        // 当相机参数改变时，更新深度材质
        viewer.addEventListener('camera-change', () => {
            if (depthMaterial && depthRender.classList.contains('checked')) {
                depthMaterial.uniforms.cameraNear.value = viewer.camera.near;
                depthMaterial.uniforms.cameraFar.value = viewer.camera.far;
                viewer.redraw();
            }
        });
    } catch (error) {
        // 错误处理
    }
}
