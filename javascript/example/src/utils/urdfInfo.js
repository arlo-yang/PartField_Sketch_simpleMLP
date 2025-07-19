/**
 * urdfInfo.js
 * 用于显示和更新URDF信息面板，通过鼠标悬停在几何体上实时显示关节信息
 * 该功能仅在segmentation模式下启用
 */

import * as THREE from 'three';

// 存储当前高亮的网格
let highlightedMeshes = [];
// 存储原始材质
const originalMaterials = new Map();
// 存储解析后的关节信息
let parsedJointInfo = null;
// 存储当前模型的关节数据
let currentJointData = null;
// 存储当前鼠标悬停的关节信息 - 设置为全局变量
window.currentHoveredJoint = null;
// 本地引用，方便代码中使用
let currentHoveredJoint = null;
// 标记是否已初始化
let isInitialized = false;
// 存储事件监听器引用，用于后续移除
let eventListeners = {};
// 存储viewer引用
let viewerRef = null;

/**
 * 根据渲染模式检查是否应该显示URDF信息面板
 * @returns {boolean} 如果当前是segmentation模式则返回true
 */
function shouldShowUrdfPanel() {
    const currentMode = window.currentRenderMode || 'default';
    return currentMode === 'segmentation';
}

/**
 * 启用URDF信息面板和相关功能
 */
function enableUrdfInfo() {
    const urdfPanel = document.getElementById('urdf-info-panel');
    if (!urdfPanel) return;
    
    // 显示面板
    if (!urdfPanel.classList.contains('visible')) {
        urdfPanel.classList.add('visible');
    }
    
    // 如果viewer存在且未初始化过，设置交互
    if (viewerRef && viewerRef.robot && !isInitialized) {
        setupJointInteraction(viewerRef);
    }
}

/**
 * 禁用URDF信息面板和相关功能
 */
function disableUrdfInfo() {
    // 隐藏面板
    const urdfPanel = document.getElementById('urdf-info-panel');
    if (urdfPanel && urdfPanel.classList.contains('visible')) {
        urdfPanel.classList.remove('visible');
    }
    
    // 清除高亮
    clearHighlights();
    
    // 移除可能的事件监听器
    if (viewerRef) {
        // 移除监听器
        Object.entries(eventListeners).forEach(([eventName, handler]) => {
            viewerRef.removeEventListener(eventName, handler);
        });
    }
}

/**
 * 处理渲染模式变化
 * @param {string} mode - 新的渲染模式
 */
function handleRenderModeChange(mode) {
    // 更新body元素的data-render-mode属性
    document.body.setAttribute('data-render-mode', mode);
    
    if (mode === 'segmentation') {
        enableUrdfInfo();
    } else {
        disableUrdfInfo();
    }
}

/**
 * 更新URDF路径显示
 * 此函数为空占位，保留以支持向后兼容
 */
function updateUrdfPath() {}

/**
 * 设置关节交互功能
 * @param {Object} viewer - URDF查看器对象
 * @returns {Promise<boolean>} - 是否成功设置关节交互
 */
async function setupJointInteraction(viewer) {
    if (!viewer) {
        return false;
    }
    
    if (!viewer.robot) {
        return false;
    }
    
    // 检查当前是否应该禁用交互
    if (shouldDisableHighlighting()) {
        return false;
    }
    
    try {
        // 获取模型中的所有关节
        const joints = viewer.robot.joints;
        if (!joints) {
            displayNoJointsMessage();
            return true;
        }
        
        const jointCount = Object.keys(joints).length;
        if (jointCount === 0) {
            displayNoJointsMessage();
            return true;
        }
        
        // 提取关节数据
        const jointData = extractJointData(viewer);
        
        // 保存关节数据
        currentJointData = jointData;
        
        // 设置鼠标交互
        setupMouseInteraction(viewer);
        
        // 显示可动关节摘要
        displayMovableJointsSummary(jointData);
        
        // 添加对urdf-manipulator元素的joint-mouseover和joint-mouseout事件的监听
        setupJointEventListeners(viewer);
        
        // 显示面板
        const urdfPanel = document.getElementById('urdf-info-panel');
        if (urdfPanel) {
            urdfPanel.classList.add('visible');
        }
        
        return true;
    } catch (error) {
        return false;
    }
}

/**
 * 设置关节事件监听器
 * @param {Object} viewer - URDF查看器对象
 */
function setupJointEventListeners(viewer) {
    // 移除可能存在的旧事件监听器
    viewer.removeEventListener('joint-mouseover', handleJointMouseOver);
    viewer.removeEventListener('joint-mouseout', handleJointMouseOut);
    
    // 添加新的事件监听器
    viewer.addEventListener('joint-mouseover', handleJointMouseOver);
    viewer.addEventListener('joint-mouseout', handleJointMouseOut);
}

/**
 * 处理关节鼠标悬停事件
 * @param {Event} event - 事件对象
 */
function handleJointMouseOver(event) {
    // 检查当前是否应该禁用高亮
    if (shouldDisableHighlighting()) {
        return;
    }
    
    const jointName = event.detail;
    
    if (!jointName) {
        return;
    }
    
    if (!currentJointData) {
        return;
    }
    
    const jointData = currentJointData[jointName];
    if (!jointData) {
        return;
    }
    
    // 高亮关节列表中的对应项
    const jointItems = document.querySelectorAll('.joint-item');
    
    jointItems.forEach(item => {
        if (item.dataset.joint === jointName) {
            item.classList.add('active');
            
            // 确保列表项可见（如果在滚动区域内）
            item.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        } else {
            item.classList.remove('active');
        }
    });
    
    // 显示关节详细信息
    displayJointDetail(jointData, jointName);
    
    // 在3D模型中高亮显示关节
    highlightJointInModel(jointName);
    
    // 更新当前悬停的关节 - 同时更新全局变量和本地变量
    currentHoveredJoint = jointName;
    window.currentHoveredJoint = jointName;
    
    // 如果当前是分段模式，触发重新渲染以更新关节颜色
    if (window.currentRenderMode === 'segmentation') {
        const viewer = document.querySelector('urdf-viewer');
        if (viewer && viewer.robot) {
            // 重新应用材质
            const applyURDFMaterialAndState = window.applyURDFMaterialAndState;
            if (typeof applyURDFMaterialAndState === 'function') {
                applyURDFMaterialAndState(viewer.robot, 'segmentation', 'original', viewer);
            } else {
                // 如果函数不可用，尝试直接重新渲染
                viewer.redraw();
            }
        }
    }
}

/**
 * 处理关节鼠标离开事件
 * @param {Event} event - 事件对象
 */
function handleJointMouseOut(event) {
    // 检查当前是否应该禁用高亮
    if (shouldDisableHighlighting()) {
        return;
    }
    
    // 移除关节列表中的高亮
    const jointItems = document.querySelectorAll('.joint-item');
    jointItems.forEach(item => {
        item.classList.remove('active');
    });
    
    // 隐藏关节详细信息
    hideJointDetail();
    
    // 清除高亮效果
    clearHighlights();
    
    // 清除当前悬停的关节 - 同时更新全局变量和本地变量
    const previousJoint = currentHoveredJoint;
    currentHoveredJoint = null;
    window.currentHoveredJoint = null;
    
    // 在所有渲染模式下重新应用材质，确保完全恢复原始状态
    const viewer = document.querySelector('urdf-viewer');
    if (viewer && viewer.robot) {
        // 获取当前渲染模式
        const currentMode = window.currentRenderMode || 'default';
        
        // 重新应用材质
        const applyURDFMaterialAndState = window.applyURDFMaterialAndState;
        if (typeof applyURDFMaterialAndState === 'function') {
            try {
                applyURDFMaterialAndState(viewer.robot, currentMode, 'original', viewer);
            } catch (error) {
                // 如果出错，尝试直接重新渲染
                viewer.redraw();
            }
        } else {
            viewer.redraw();
        }
    }
}

/**
 * 提取关节数据
 * @param {Object} viewer - URDF查看器对象
 * @returns {Object} - 关节数据对象
 */
function extractJointData(viewer) {
    if (!viewer || !viewer.robot || !viewer.robot.joints) {
        return {};
    }
    
    const joints = viewer.robot.joints;
    const jointData = {};
    
    // 遍历所有关节，提取信息
    for (const [jointName, joint] of Object.entries(joints)) {
        // 检查关节对象
        if (!joint) {
            continue;
        }
        
        // 获取关节类型
        const jointType = joint.jointType;
        
        // 如果仍然有continuous类型，在UI层面强制转换为fixed
        let displayJointType = jointType;
        if (displayJointType === 'continuous') {
            displayJointType = 'fixed';
        }
        
        // 检查prismatic关节的范围，如果范围小于等于0.15，在UI层面强制转换为fixed
        if (displayJointType === 'prismatic') {
            const lower = joint.limit?.lower !== undefined ? joint.limit.lower : 0;
            const upper = joint.limit?.upper !== undefined ? joint.limit.upper : 0;
            if (upper - lower <= 0.15) {
                displayJointType = 'fixed';
            }
        }
        
        // 获取关节轴向
        let axis;
        if (joint.axis) {
            axis = new THREE.Vector3().copy(joint.axis);
        } else {
            axis = new THREE.Vector3(0, 0, 0);
        }
        
        // 获取关节限制
        const limits = {
            lower: joint.limit?.lower !== undefined ? joint.limit.lower : 'none',
            upper: joint.limit?.upper !== undefined ? joint.limit.upper : 'none',
            effort: joint.limit?.effort !== undefined ? joint.limit.effort : 'none',
            velocity: joint.limit?.velocity !== undefined ? joint.limit.velocity : 'none'
        };
        
        // 获取关节当前值
        let currentValue;
        try {
            if (joint.angle !== undefined) {
                currentValue = joint.angle;
            } else if (typeof joint.getJointValue === 'function') {
                currentValue = joint.getJointValue();
            } else if (joint.value !== undefined) {
                currentValue = joint.value;
            }
        } catch (error) {
            // 忽略错误
        }
        
        // 获取父链接和子链接
        let parentLink = 'unknown';
        let childLink = 'unknown';
        
        // 尝试从parent获取父链接
        if (joint.parent && joint.parent.isURDFLink) {
            parentLink = joint.parent.urdfName || joint.parent.name || 'unknown';
        }
        
        // 尝试从children获取子链接
        if (joint.children && joint.children.length > 0) {
            for (const child of joint.children) {
                if (child.isURDFLink) {
                    childLink = child.urdfName || child.name || 'unknown';
                    break;
                }
            }
        }
        
        // 尝试从属性获取链接名称
        if (joint.parentName) {
            parentLink = joint.parentName;
        }
        
        if (joint.name) {
            childLink = joint.name;
        }
        
        // 获取关节位置
        const position = new THREE.Vector3();
        if (joint.matrixWorld) {
            position.setFromMatrixPosition(joint.matrixWorld);
        }
        
        // 创建关节信息对象
        jointData[jointName] = {
            type: displayJointType, // 使用转换后的关节类型
            axis: {
                x: parseFloat(axis.x.toFixed(4)),
                y: parseFloat(axis.y.toFixed(4)),
                z: parseFloat(axis.z.toFixed(4))
            },
            limits: limits,
            parentLink: parentLink,
            childLink: childLink,
            position: {
                x: parseFloat(position.x.toFixed(4)),
                y: parseFloat(position.y.toFixed(4)),
                z: parseFloat(position.z.toFixed(4))
            },
            movable: displayJointType !== 'fixed', // 使用转换后的关节类型判断是否可动
            currentValue: currentValue
        };
    }
    
    return jointData;
}

/**
 * 显示可动关节摘要
 * @param {Object} jointData - 关节数据
 */
function displayMovableJointsSummary(jointData) {
    const jointInfoDisplay = document.getElementById('joint-info-display');
    if (!jointInfoDisplay) return;
    
    // 筛选出可动关节
    const movableJoints = Object.entries(jointData).filter(([_, joint]) => joint.movable);
    
    if (movableJoints.length === 0) {
        jointInfoDisplay.innerHTML = `
            <div class="joint-info-placeholder">
                No movable joints found in this model.<br>
                All parts are fixed.
            </div>
        `;
        return;
    }
    
    // 创建可动关节摘要
    let summaryHTML = `
        <div class="joint-summary">
            <h3>Movable Joints (${movableJoints.length})</h3>
            <ul class="joint-list">
    `;
    
    movableJoints.forEach(([jointName, joint]) => {
        summaryHTML += `
            <li class="joint-item" data-joint="${jointName}">
                <span class="joint-name">${jointName}</span>
                <span class="joint-type">(${joint.type})</span>
            </li>
        `;
    });
    
    summaryHTML += `
            </ul>
            <div class="joint-hover-instruction">Hover over model parts to see detailed joint information</div>
        </div>
        <div class="joint-detail-container">
            <div class="joint-detail-placeholder">
                Select a joint to see details
            </div>
        </div>
    `;
    
    jointInfoDisplay.innerHTML = summaryHTML;
    
    // 为关节列表项添加交互
    const jointItems = jointInfoDisplay.querySelectorAll('.joint-item');
    jointItems.forEach(item => {
        item.addEventListener('mouseenter', () => {
            const jointName = item.dataset.joint;
            if (jointName && jointData[jointName]) {
                displayJointDetail(jointData[jointName], jointName);
                highlightJointInModel(jointName);
            }
        });
        
        item.addEventListener('mouseleave', () => {
            clearHighlights();
            hideJointDetail();
        });
    });
}

/**
 * 显示关节详细信息
 * @param {Object} joint - 关节数据
 * @param {string} jointName - 关节名称
 */
function displayJointDetail(joint, jointName) {
    const jointDetailContainer = document.querySelector('.joint-detail-container');
    if (!jointDetailContainer) return;
    
    // 格式化关节限制
    const formatLimit = (limit, isRotational = false) => {
        if (limit === 'none') return 'None';
        if (typeof limit === 'number') {
            const radianValue = limit.toFixed(4);
            // 如果是旋转关节，添加角度显示
            if (isRotationalJoint) {
                const degrees = (limit * 180 / Math.PI).toFixed(2);
                return `${radianValue} rad (${degrees}°)`;
            }
            return radianValue;
        }
        return limit;
    };
    
    // 判断是否为旋转类型关节
    const isRotationalJoint = joint.type === 'revolute' || joint.type === 'continuous';
    
    // 创建详细信息HTML
    let detailHTML = `
        <div class="joint-detail">
            <h4>${jointName}</h4>
            <table class="joint-detail-table">
                <tr>
                    <td class="detail-label">Type:</td>
                    <td class="detail-value">${joint.type}</td>
                </tr>
                <tr>
                    <td class="detail-label">Axis:</td>
                    <td class="detail-value">X: ${joint.axis.x}, Y: ${joint.axis.y}, Z: ${joint.axis.z}</td>
                </tr>
    `;
    
    // 如果是可动关节，只显示上下限制
    if (joint.movable) {
        detailHTML += `
                <tr>
                    <td class="detail-label">Limits:</td>
                    <td class="detail-value">
                        <table class="limits-table">
                            <tr>
                                <td>Lower:</td>
                                <td>${formatLimit(joint.limits.lower, isRotationalJoint)}</td>
                            </tr>
                            <tr>
                                <td>Upper:</td>
                                <td>${formatLimit(joint.limits.upper, isRotationalJoint)}</td>
                            </tr>
                        </table>
                    </td>
                </tr>
        `;
    }
    
    detailHTML += `
            </table>
        </div>
    `;
    
    jointDetailContainer.innerHTML = detailHTML;
}

/**
 * 隐藏关节详细信息
 */
function hideJointDetail() {
    const jointDetailContainer = document.querySelector('.joint-detail-container');
    if (!jointDetailContainer) return;
    
    jointDetailContainer.innerHTML = `
        <div class="joint-detail-placeholder">
            Select a joint to see details
        </div>
    `;
}

/**
 * 显示没有关节的消息
 */
function displayNoJointsMessage() {
    const jointInfoDisplay = document.getElementById('joint-info-display');
    if (!jointInfoDisplay) return;
    
    jointInfoDisplay.innerHTML = `
        <div class="joint-info-placeholder">
            <h3>No Joints Found</h3>
            <p>This model does not contain any movable joints. All parts are fixed.</p>
            <p>Model: ${window.currentUrdfPath || 'Unknown'}</p>
        </div>
    `;
}

/**
 * 设置鼠标交互
 * @param {Object} viewer - URDF查看器对象
 */
function setupMouseInteraction(viewer) {
    if (!viewer || !viewer.robot) {
        return;
    }
    
    try {
        // 清除之前可能存在的事件监听器
        viewer.robot.traverse(child => {
            if (child.isMesh) {
                if (child._mouseoverHandler) {
                    child.removeEventListener('mouseover', child._mouseoverHandler);
                }
                if (child._mouseoutHandler) {
                    child.removeEventListener('mouseout', child._mouseoutHandler);
                }
            }
        });
        
        // 检查是否有网格
        let hasMeshes = false;
        viewer.robot.traverse(child => {
            if (child.isMesh) {
                hasMeshes = true;
            }
        });
        
        if (!hasMeshes) {
            return;
        }
        
        // 为模型中的每个网格添加鼠标事件
        viewer.robot.traverse(child => {
            if (child.isMesh) {
                // 保存原始材质
                if (!originalMaterials.has(child.uuid)) {
                    // 深度克隆材质，确保完全复制所有属性
                    const originalMaterial = child.material.clone();
                    originalMaterials.set(child.uuid, originalMaterial);
                    
                    // 同时在 userData 中保存一份，作为备份
                    if (!child.userData.originalMaterial) {
                        child.userData.originalMaterial = originalMaterial;
                    }
                }
                
                // 添加鼠标悬停事件
                child._mouseoverHandler = () => {
                    handleMeshHover(child, viewer);
                };
                
                // 检查是否支持addEventListener
                if (typeof child.addEventListener === 'function') {
                    child.addEventListener('mouseover', child._mouseoverHandler);
                } else {
                    // 尝试使用Three.js的事件系统
                    if (typeof THREE.EventDispatcher !== 'undefined') {
                        THREE.EventDispatcher.prototype.addEventListener.call(child, 'mouseover', child._mouseoverHandler);
                    }
                }
                
                // 添加鼠标离开事件
                child._mouseoutHandler = () => {
                    clearHighlights();
                    // 如果当前没有显示详细信息，则恢复到可动关节摘要
                    if (!currentHoveredJoint) {
                        hideJointDetail();
                    }
                };
                
                // 检查是否支持addEventListener
                if (typeof child.addEventListener === 'function') {
                    child.addEventListener('mouseout', child._mouseoutHandler);
                } else {
                    // 尝试使用Three.js的事件系统
                    if (typeof THREE.EventDispatcher !== 'undefined') {
                        THREE.EventDispatcher.prototype.addEventListener.call(child, 'mouseout', child._mouseoutHandler);
                    }
                }
                
                // 确保网格可以接收鼠标事件
                child.userData.selectable = true;
            }
        });
    } catch (error) {
        // 错误处理
    }
}

/**
 * 处理网格悬停事件
 * @param {Object} mesh - 悬停的网格
 * @param {Object} viewer - URDF查看器对象
 */
function handleMeshHover(mesh, viewer) {
    if (!mesh || !viewer || !viewer.robot) {
        return;
    }
    
    // 检查当前是否应该禁用高亮
    if (shouldDisableHighlighting()) {
        return;
    }
    
    // 清除之前的高亮
    clearHighlights();
    
    // 创建高亮材质
    const highlightMaterial = new THREE.MeshPhongMaterial({
        color: mesh.material.color ? mesh.material.color.clone() : new THREE.Color(0xff0000),
        emissive: new THREE.Color(0xff0000),
        emissiveIntensity: 0.5,
        shininess: 30,
        transparent: true,
        opacity: 0.8
    });
    
    // 应用高亮材质
    if (!originalMaterials.has(mesh.uuid)) {
        originalMaterials.set(mesh.uuid, mesh.material.clone());
    }
    mesh.material = highlightMaterial;
    
    // 添加到高亮列表
    highlightedMeshes.push(mesh);
    
    // 查找最近的关节
    let currentObject = mesh;
    let nearestJoint = null;
    let jointName = '';
    
    // 向上遍历对象层次结构，寻找最近的关节
    while (currentObject && !nearestJoint) {
        if (currentObject.isURDFJoint) {
            nearestJoint = currentObject;
            jointName = currentObject.urdfName || currentObject.name;
            break;
        }
        
        // 如果当前对象有父对象，继续向上查找
        currentObject = currentObject.parent;
    }
    
    // 如果通过对象层次结构没有找到关节，尝试通过链接名称查找
    if (!nearestJoint) {
        // 查找网格所属的链接
        let linkName = '';
        let linkObject = null;
        
        currentObject = mesh;
        while (currentObject && !linkObject) {
            if (currentObject.isURDFLink) {
                linkObject = currentObject;
                linkName = currentObject.urdfName || currentObject.name;
                break;
            }
            currentObject = currentObject.parent;
        }
        
        // 如果找到了链接，查找与该链接相关的关节
        if (linkName) {
            // 首先检查是否有currentJointData
            if (currentJointData) {
                for (const [name, joint] of Object.entries(currentJointData)) {
                    if (joint.childLink === linkName) {
                        nearestJoint = joint;
                        jointName = name;
                        break;
                    } else if (joint.parentLink === linkName && !nearestJoint) {
                        // 如果找到了父链接匹配，先保存，但继续寻找可能的子链接匹配
                        nearestJoint = joint;
                        jointName = name;
                    }
                }
            } else {
                // 如果没有currentJointData，尝试直接从robot.joints查找
                for (const [name, joint] of Object.entries(viewer.robot.joints)) {
                    // 检查关节的父链接
                    let parentLink = null;
                    if (joint.parent && joint.parent.isURDFLink) {
                        parentLink = joint.parent.urdfName || joint.parent.name;
                    }
                    
                    // 检查关节的子链接
                    let childLink = null;
                    if (joint.children) {
                        for (const child of joint.children) {
                            if (child.isURDFLink) {
                                childLink = child.urdfName || child.name;
                                break;
                            }
                        }
                    }
                    
                    if (childLink === linkName) {
                        nearestJoint = joint;
                        jointName = name;
                        break;
                    } else if (parentLink === linkName && !nearestJoint) {
                        // 如果找到了父链接匹配，先保存，但继续寻找可能的子链接匹配
                        nearestJoint = joint;
                        jointName = name;
                    }
                }
            }
        }
    }
    
    // 如果仍然没有找到关节，尝试遍历所有关节，查找与网格相关的关节
    if (!nearestJoint) {
        // 获取网格的世界坐标
        const meshPosition = new THREE.Vector3();
        mesh.getWorldPosition(meshPosition);
        
        let closestDistance = Infinity;
        
        // 遍历所有关节，找到最近的关节
        for (const [name, joint] of Object.entries(viewer.robot.joints)) {
            // 获取关节的世界坐标
            const jointPosition = new THREE.Vector3();
            joint.getWorldPosition(jointPosition);
            
            // 计算距离
            const distance = meshPosition.distanceTo(jointPosition);
            
            if (distance < closestDistance) {
                closestDistance = distance;
                nearestJoint = joint;
                jointName = name;
            }
        }
        
        // 如果找到了最近的关节，但距离太远，可能不是相关的关节
        if (closestDistance > 1.0) {
            // 距离太远，不处理
        }
    }
    
    // 如果找到相关关节，显示详细信息
    if (nearestJoint) {
        currentHoveredJoint = jointName;
        
        // 高亮关节列表中的对应项
        const jointItems = document.querySelectorAll('.joint-item');
        
        jointItems.forEach(item => {
            if (item.dataset.joint === jointName) {
                item.classList.add('active');
                
                // 确保列表项可见（如果在滚动区域内）
                item.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            } else {
                item.classList.remove('active');
            }
        });
        
        // 获取关节数据
        let jointData = null;
        
        if (nearestJoint.isURDFJoint) {
            // 如果是直接从对象层次结构找到的关节
            
            // 获取关节轴向
            let axis = new THREE.Vector3(0, 0, 0);
            if (nearestJoint.axis) {
                axis = new THREE.Vector3().copy(nearestJoint.axis);
            }
            
            // 获取关节限制
            const limits = {
                lower: nearestJoint.limit?.lower !== undefined ? nearestJoint.limit.lower : 'none',
                upper: nearestJoint.limit?.upper !== undefined ? nearestJoint.limit.upper : 'none',
                effort: nearestJoint.limit?.effort !== undefined ? nearestJoint.limit.effort : 'none',
                velocity: nearestJoint.limit?.velocity !== undefined ? nearestJoint.limit.velocity : 'none'
            };
            
            // 获取父链接和子链接
            let parentLink = 'unknown';
            let childLink = 'unknown';
            
            if (nearestJoint.parent && nearestJoint.parent.isURDFLink) {
                parentLink = nearestJoint.parent.urdfName || nearestJoint.parent.name || 'unknown';
            }
            
            if (nearestJoint.children) {
                for (const child of nearestJoint.children) {
                    if (child.isURDFLink) {
                        childLink = child.urdfName || child.name || 'unknown';
                        break;
                    }
                }
            }
            
            // 获取关节位置
            const position = new THREE.Vector3();
            if (nearestJoint.matrixWorld) {
                position.setFromMatrixPosition(nearestJoint.matrixWorld);
            }
            
            jointData = {
                type: nearestJoint.jointType,
                axis: {
                    x: parseFloat(axis.x.toFixed(4)),
                    y: parseFloat(axis.y.toFixed(4)),
                    z: parseFloat(axis.z.toFixed(4))
                },
                limits: limits,
                parentLink: parentLink,
                childLink: childLink,
                position: {
                    x: parseFloat(position.x.toFixed(4)),
                    y: parseFloat(position.y.toFixed(4)),
                    z: parseFloat(position.z.toFixed(4))
                },
                movable: nearestJoint.jointType !== 'fixed'
            };
        } else if (currentJointData && currentJointData[jointName]) {
            // 如果是从currentJointData中找到的关节
            jointData = currentJointData[jointName];
        } else {
            // 如果没有找到关节数据，创建一个基本的数据结构
            jointData = {
                type: 'unknown',
                axis: { x: 0, y: 0, z: 0 },
                limits: { lower: 'none', upper: 'none', effort: 'none', velocity: 'none' },
                parentLink: 'unknown',
                childLink: 'unknown',
                position: { x: 0, y: 0, z: 0 },
                movable: false
            };
        }
        
        // 显示关节详细信息
        displayJointDetail(jointData, jointName);
        
        // 在3D模型中高亮显示关节
        highlightJointInModel(jointName);
    } else {
        currentHoveredJoint = null;
        hideJointDetail();
    }
}

/**
 * 在3D模型中高亮显示关节
 * @param {string} jointName - 关节名称
 */
function highlightJointInModel(jointName) {
    // 检查当前是否应该禁用高亮
    if (shouldDisableHighlighting()) {
        return;
    }
    
    const viewer = document.querySelector('urdf-viewer');
    if (!viewer || !viewer.robot || !viewer.robot.joints || !viewer.scene) {
        return;
    }
    
    const joint = viewer.robot.joints[jointName];
    if (!joint) {
        return;
    }
    
    // 获取关节的子链接
    let childLink = null;
    
    // 查找关节的子链接
    if (joint.children && joint.children.length > 0) {
        for (const child of joint.children) {
            if (child.isURDFLink) {
                childLink = child;
                break;
            }
        }
    }
    
    if (!childLink) {
        // 如果没有直接的子链接，尝试查找所有链接
        for (const linkName in viewer.robot.links) {
            const link = viewer.robot.links[linkName];
            
            // 检查链接的父对象是否是当前关节
            let parent = link.parent;
            while (parent) {
                if (parent === joint) {
                    childLink = link;
                    break;
                }
                parent = parent.parent;
            }
            
            if (childLink) break;
        }
    }
    
    // 高亮子链接下的所有网格
    let meshCount = 0;
    
    if (childLink) {
        childLink.traverse(child => {
            if (child.isMesh) {
                meshCount++;
                
                // 保存原始材质
                if (!originalMaterials.has(child.uuid)) {
                    // 深度克隆材质，确保完全复制所有属性
                    const originalMaterial = child.material.clone();
                    originalMaterials.set(child.uuid, originalMaterial);
                    
                    // 同时在 userData 中保存一份，作为备份
                    if (!child.userData.originalMaterial) {
                        child.userData.originalMaterial = originalMaterial;
                    }
                }
                
                // 创建高亮材质
                const highlightMaterial = new THREE.MeshPhongMaterial({
                    color: child.material.color ? child.material.color.clone() : new THREE.Color(0xffffff),
                    emissive: new THREE.Color(0xffffff),
                    emissiveIntensity: 0.5,
                    shininess: 30,
                    transparent: true,
                    opacity: 0.8
                });
                
                // 应用高亮材质
                child.material = highlightMaterial;
                
                // 添加到高亮列表
                highlightedMeshes.push(child);
            }
        });
    } else {
        // 如果没有找到子链接，尝试直接高亮关节对象
        joint.traverse(child => {
            if (child.isMesh) {
                meshCount++;
                
                // 保存原始材质
                if (!originalMaterials.has(child.uuid)) {
                    // 深度克隆材质，确保完全复制所有属性
                    const originalMaterial = child.material.clone();
                    originalMaterials.set(child.uuid, originalMaterial);
                    
                    // 同时在 userData 中保存一份，作为备份
                    if (!child.userData.originalMaterial) {
                        child.userData.originalMaterial = originalMaterial;
                    }
                }
                
                // 创建高亮材质
                const highlightMaterial = new THREE.MeshPhongMaterial({
                    color: child.material.color ? child.material.color.clone() : new THREE.Color(0xffffff),
                    emissive: new THREE.Color(0xffffff),
                    emissiveIntensity: 0.5,
                    shininess: 30,
                    transparent: true,
                    opacity: 0.8
                });
                
                // 应用高亮材质
                child.material = highlightMaterial;
                
                // 添加到高亮列表
                highlightedMeshes.push(child);
            }
        });
        
        if (meshCount === 0) {
            // 如果关节对象下没有网格，尝试高亮显示关节的父链接
            if (joint.parent && joint.parent.isURDFLink) {
                const parentLink = joint.parent;
                
                parentLink.traverse(child => {
                    if (child.isMesh) {
                        meshCount++;
                        
                        // 保存原始材质
                        if (!originalMaterials.has(child.uuid)) {
                            // 深度克隆材质，确保完全复制所有属性
                            const originalMaterial = child.material.clone();
                            originalMaterials.set(child.uuid, originalMaterial);
                            
                            // 同时在 userData 中保存一份，作为备份
                            if (!child.userData.originalMaterial) {
                                child.userData.originalMaterial = originalMaterial;
                            }
                        }
                        
                        // 创建高亮材质
                        const highlightMaterial = new THREE.MeshPhongMaterial({
                            color: child.material.color ? child.material.color.clone() : new THREE.Color(0xffffff),
                            emissive: new THREE.Color(0xffffff),
                            emissiveIntensity: 0.5,
                            shininess: 30,
                            transparent: true,
                            opacity: 0.8
                        });
                        
                        // 应用高亮材质
                        child.material = highlightMaterial;
                        
                        // 添加到高亮列表
                        highlightedMeshes.push(child);
                    }
                });
            }
        }
    }
    
    // 如果是可动关节，显示运动轴
    if (joint.jointType !== 'fixed' && joint.axis) {
        try {
            // 创建箭头辅助对象显示运动轴
            const origin = new THREE.Vector3();
            joint.getWorldPosition(origin);
            
            // 获取世界坐标系中的轴向
            const direction = new THREE.Vector3().copy(joint.axis);
            direction.applyQuaternion(joint.getWorldQuaternion(new THREE.Quaternion()));
            direction.normalize();
            
            const length = 0.5; // 箭头长度
            
            // 创建箭头
            const arrowHelper = new THREE.ArrowHelper(
                direction,
                origin,
                length,
                0x0000ff, // 蓝色
                length * 0.2, // 箭头头部长度
                length * 0.1  // 箭头头部宽度
            );
            
            // 添加到场景
            viewer.scene.add(arrowHelper);
            
            // 保存箭头，以便后续清除
            highlightedMeshes.push(arrowHelper);
        } catch (error) {
            // 忽略错误
        }
    }
    
    // 重新渲染场景
    if (viewer.redraw && typeof viewer.redraw === 'function') {
        try {
            viewer.redraw();
        } catch (error) {
            // 尝试使用renderer直接渲染
            if (viewer.renderer && viewer.camera) {
                try {
                    viewer.renderer.render(viewer.scene, viewer.camera);
                } catch (error) {
                    // 忽略错误
                }
            }
        }
    } else {
        // 尝试使用renderer直接渲染
        if (viewer.renderer && viewer.camera) {
            try {
                viewer.renderer.render(viewer.scene, viewer.camera);
            } catch (error) {
                // 忽略错误
            }
        }
    }
}

/**
 * 清除所有高亮效果
 */
export function clearHighlights() {
    const viewer = document.querySelector('urdf-viewer');
    if (!viewer) {
        return;
    }
    
    // 处理高亮对象
    highlightedMeshes.forEach(obj => {
        if (!obj) {
            return;
        }
        
        try {
            if (obj instanceof THREE.ArrowHelper) {
                // 如果是箭头辅助对象，从场景中移除
                if (obj.parent) {
                    obj.parent.remove(obj);
                }
                
                // 释放资源
                if (obj.dispose) {
                    obj.dispose();
                }
            } else if (obj.isMesh) {
                // 如果是网格，恢复原始材质
                const originalMaterial = originalMaterials.get(obj.uuid);
                if (originalMaterial) {
                    obj.material = originalMaterial;
                } else {
                    // 如果找不到原始材质，尝试使用userData中的originalMaterial
                    if (obj.userData && obj.userData.originalMaterial) {
                        obj.material = obj.userData.originalMaterial;
                    }
                }
                
                // 确保材质已正确应用
                if (obj.material) {
                    // 强制更新材质
                    obj.material.needsUpdate = true;
                }
            }
        } catch (error) {
            // 忽略错误
        }
    });
    
    // 清空高亮列表
    highlightedMeshes = [];
    
    // 清空原始材质映射
    originalMaterials.clear();
    
    // 取消关节列表中的高亮
    try {
        const jointItems = document.querySelectorAll('.joint-item');
        jointItems.forEach(item => {
            if (item.classList.contains('active')) {
                item.classList.remove('active');
            }
        });
    } catch (error) {
        // 忽略错误
    }
    
    // 重新渲染场景
    try {
        if (viewer.redraw && typeof viewer.redraw === 'function') {
            viewer.redraw();
        } else {
            // 尝试使用renderer直接渲染
            if (viewer.renderer && viewer.camera) {
                viewer.renderer.render(viewer.scene, viewer.camera);
            }
        }
    } catch (error) {
        // 忽略错误
    }
}

/**
 * 调整面板位置，确保不重叠
 */
function adjustPanelPositions() {
    const urdfPanel = document.getElementById('urdf-info-panel');
    
    // 设置面板顶部位置
    if (urdfPanel) {
        urdfPanel.style.top = '20px';
    }
    
    // 触发全局事件，通知其他面板调整位置
    const event = new CustomEvent('panels-repositioned');
    window.dispatchEvent(event);
}

/**
 * 检查当前是否应该禁用鼠标高亮
 * @returns {boolean} 如果当前模式应该禁用高亮，则返回true
 */
function shouldDisableHighlighting() {
    return window.currentRenderMode && window.currentRenderMode.startsWith('arrow');
} 

/**
 * 初始化URDF信息显示功能
 * @param {Object} viewer - URDF查看器对象
 */
export function initUrdfInfo(viewer) {
    // 保存viewer引用
    viewerRef = viewer;
    
    // 检查viewer是否存在
    if (!viewer) {
        return;
    }
    
    // 获取DOM元素
    const urdfPanel = document.getElementById('urdf-info-panel');
    const urdfToggle = document.getElementById('urdf-toggle');
    const urdfContent = document.getElementById('urdf-info-content');
    const urdfHeader = document.querySelector('.urdf-info-header');
    const urdfViewerContainer = document.querySelector('.urdf-viewer-container');
    const loadJointInfoButton = document.getElementById('load-urdf-content');
    
    // 检查当前渲染模式，只有在segmentation模式下才继续初始化
    if (!shouldShowUrdfPanel()) {
        // 确保面板隐藏
        if (urdfPanel) {
            urdfPanel.classList.remove('visible');
        }
        
        // 注册渲染模式变化监听，用于后续可能的模式切换
        window.addEventListener('renderModeChanged', (e) => {
            const mode = e.detail?.mode || 'default';
            handleRenderModeChange(mode);
        });
        
        return; // 退出函数，不继续初始化
    }
    
    // 如果是segmentation模式，继续初始化
    
    // 更新UI标题
    if (urdfHeader) {
        urdfHeader.innerHTML = 'Joint Information <span id="urdf-toggle" class="info-toggle">▼</span>';
    }
    
    // 确保初始状态是折叠的
    if (urdfContent && urdfToggle) {
        urdfContent.classList.add('collapsed');
        urdfToggle.classList.add('collapsed');
        urdfToggle.textContent = '▶';
    }
    
    // 设置折叠/展开功能
    if (urdfToggle && urdfContent && urdfHeader) {
        // 为整个标题添加点击事件
        urdfHeader.addEventListener('click', (e) => {
            // 如果点击的是按钮或其他控件，则不触发折叠/展开
            if (e.target.tagName === 'BUTTON' || e.target.closest('button')) {
                return;
            }
            
            // 如果点击的是info-item或其子元素，不触发折叠/展开
            if (e.target.closest('.info-item')) {
                return;
            }
            
            urdfContent.classList.toggle('collapsed');
            urdfToggle.classList.toggle('collapsed');
            urdfToggle.textContent = urdfContent.classList.contains('collapsed') ? '▶' : '▼';
            
            // 调整面板位置
            adjustPanelPositions();
            
            setTimeout(() => {
                adjustPanelPositions();
            }, 300);
        });
        
        // 为折叠图标添加单独的点击事件
        urdfToggle.addEventListener('click', (e) => {
            e.stopPropagation();
            urdfContent.classList.toggle('collapsed');
            urdfToggle.classList.toggle('collapsed');
            urdfToggle.textContent = urdfContent.classList.contains('collapsed') ? '▶' : '▼';
            
            adjustPanelPositions();
            
            setTimeout(() => {
                adjustPanelPositions();
            }, 300);
        });
    }
    
    // 更新URDF内容区域
    if (urdfViewerContainer) {
        urdfViewerContainer.innerHTML = `
            <div id="joint-info-display" class="joint-info-display">
                <div class="joint-info-placeholder">
                    Hover over model parts to see joint information
                </div>
            </div>
        `;
    }
    
    // 设置按钮功能
    if (loadJointInfoButton) {
        loadJointInfoButton.textContent = 'Enable Joint Interaction';
        
        loadJointInfoButton.addEventListener('click', (e) => {
            e.stopPropagation();
            
            loadJointInfoButton.disabled = true;
            loadJointInfoButton.textContent = 'Enabling...';
            
            setupJointInteraction(viewer)
                .then(success => {
                    if (success) {
                        loadJointInfoButton.textContent = 'Refresh Interaction';
                        loadJointInfoButton.classList.add('success');
                        loadJointInfoButton.classList.remove('error');
                    } else {
                        loadJointInfoButton.textContent = 'Failed to Enable';
                        loadJointInfoButton.classList.add('error');
                    }
                    
                    loadJointInfoButton.disabled = false;
                    
                    if (urdfPanel) {
                        urdfPanel.classList.add('visible');
                    }
                    
                    adjustPanelPositions();
                })
                .catch(error => {
                    loadJointInfoButton.textContent = 'Failed to Enable';
                    loadJointInfoButton.classList.add('error');
                    loadJointInfoButton.disabled = false;
                });
        });
    }
    
    // 设置面板重定位
    window.addEventListener('panels-repositioned', () => {
        setTimeout(() => {
            adjustPanelPositions();
        }, 10);
    });
    
    // 设置事件监听器，保存引用以便后续移除
    eventListeners = {
        'urdf-processed': () => {
            if (shouldShowUrdfPanel()) {
                if (loadJointInfoButton && !loadJointInfoButton.disabled) {
                    setTimeout(() => {
                        loadJointInfoButton.click();
                    }, 500);
                }
            }
        },
        'joint-mouseover': handleJointMouseOver,
        'joint-mouseout': handleJointMouseOut,
        'model-geometry-update': () => {
            if (shouldShowUrdfPanel()) {
                updateUrdfPath();
            }
        }
    };
    
    // 添加事件监听器
    Object.entries(eventListeners).forEach(([eventName, handler]) => {
        viewer.addEventListener(eventName, handler);
    });
    
    // 监听渲染模式变化
    window.addEventListener('renderModeChanged', (e) => {
        const mode = e.detail?.mode || 'default';
        handleRenderModeChange(mode);
    });
    
    // 如果当前已经有robot，立即设置交互
    if (viewer && viewer.robot && shouldShowUrdfPanel()) {
        if (urdfPanel) {
            urdfPanel.classList.add('visible');
        }
        
        setupJointInteraction(viewer)
            .then(success => {
                if (loadJointInfoButton) {
                    if (success) {
                        loadJointInfoButton.textContent = 'Refresh Interaction';
                        loadJointInfoButton.classList.add('success');
                    } else {
                        loadJointInfoButton.textContent = 'Failed to Enable';
                        loadJointInfoButton.classList.add('error');
                    }
                }
            })
            .catch(error => {
                // 错误处理
            });
    }
    
    // 设置初始化完成标记
    isInitialized = true;
} 