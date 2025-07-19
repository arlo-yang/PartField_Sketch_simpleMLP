// 管理材质（Materials）
// 处理视觉效果（Visual Effects）
// 控制渲染状态（Render States）
// 处理模型外观（Appearance）

import * as THREE from 'three';
import { loadArrowModel, applyArrowMaterial, removeArrowObject, removeArrowContainer, handleArrowStateChange } from './arrow.js';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { initialModels } from '../models-config.js';
import { clearHighlights } from './urdfInfo.js';

// Materials 部分
// ===============

// 普通材质
export const normalMaterial = new THREE.MeshNormalMaterial({
    flatShading: false,
    transparent: true,
    opacity: 1
});

// 深度图材质
export let depthMaterial = new THREE.ShaderMaterial({
    uniforms: {
        cameraNear: { value: 0.1 },
        cameraFar: { value: 1000.0 },
        sceneDepthMin: { value: 0.0 },
        sceneDepthMax: { value: 1000.0 }
    },
    vertexShader: `
        varying float vViewZ;
        void main() {
            vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
            gl_Position = projectionMatrix * mvPosition;
            vViewZ = -mvPosition.z;
        }
    `,
    fragmentShader: `
        uniform float cameraNear;
        uniform float cameraFar;
        uniform float sceneDepthMin;
        uniform float sceneDepthMax;
        varying float vViewZ;
        
        void main() {
            if(vViewZ > cameraFar) {
                gl_FragColor = vec4(1.0);
                return;
            }
            
            // 计算线性深度，并反转值（1 - depth）使得近处更亮
            float linearDepth = (vViewZ - sceneDepthMin) / (sceneDepthMax - sceneDepthMin);
            linearDepth = 1.0 - clamp(linearDepth, 0.0, 1.0);  // 反转深度值
            
            float gamma = 0.5;
            float depth = pow(linearDepth, gamma);
            
            gl_FragColor = vec4(vec3(depth), 1.0);
        }
    `,
    side: THREE.DoubleSide
});

// 线框材质
export const wireframeMaterial = new THREE.MeshBasicMaterial({
    wireframe: true,
    color: 0xffffff,
    wireframeLinewidth: 1
});

// segmentation
export const segmentationStaticMaterial = new THREE.MeshBasicMaterial({ 
    color: 0x000000  // 静态部分用黑色
});

export const segmentationMovableMaterial = new THREE.MeshBasicMaterial({ 
    color: 0xffffff  // 可动部分用白色
});



// OBJ 模型材质
export const objMaterial = new THREE.MeshPhongMaterial({
    color: 0xcccccc,
    shininess: 30,
    flatShading: false
});

// Render 功能部分
// ==============

function computeSceneDepthRange(scene, camera) {
    if (!scene || !camera) {
        return { min: 0.1, max: 2000.0 };
    }

    let minDepth = Infinity;
    let maxDepth = -Infinity;
    let meshCount = 0;
    
    try {
        const tempVector = new THREE.Vector3();
        const cameraPosition = camera.position;
        
        scene.traverse((object) => {
            if (object.isMesh && object.geometry) {
                meshCount++;
                const geometry = object.geometry;
                
                if (!geometry.boundingBox) {
                    geometry.computeBoundingBox();
                }
                
                if (!geometry.boundingBox) {
                    return;
                }
                
                const box = geometry.boundingBox;
                const corners = [
                    new THREE.Vector3(box.min.x, box.min.y, box.min.z),
                    new THREE.Vector3(box.min.x, box.min.y, box.max.z),
                    new THREE.Vector3(box.min.x, box.max.y, box.min.z),
                    new THREE.Vector3(box.min.x, box.max.y, box.max.z),
                    new THREE.Vector3(box.max.x, box.min.y, box.min.z),
                    new THREE.Vector3(box.max.x, box.min.y, box.max.z),
                    new THREE.Vector3(box.max.x, box.max.y, box.min.z),
                    new THREE.Vector3(box.max.x, box.max.y, box.max.z)
                ];
                
                corners.forEach(corner => {
                    tempVector.copy(corner);
                    object.localToWorld(tempVector);  // 转换到世界坐标
                    const depth = tempVector.distanceTo(cameraPosition);
                    minDepth = Math.min(minDepth, depth);
                    maxDepth = Math.max(maxDepth, depth);
                });
            }
        });
        
        if (minDepth === Infinity || maxDepth === -Infinity) {
            minDepth = 0.1;
            maxDepth = 1000.0;
        } else {
            // 添加余量并确保最小值
            minDepth = Math.max(0.1, minDepth * 0.9);
            maxDepth = maxDepth * 1.1;
        }
    } catch (error) {
        minDepth = 0.1;
        maxDepth = 1000.0;
    }
    return { min: minDepth, max: maxDepth };
}

// 处理相对路径，将其转换为绝对URL
function convertToAbsolutePath(relativePath) {
    // 如果已经是绝对路径，直接返回
    if (relativePath.startsWith('http')) {
        return relativePath;
    }
    
    // 从相对路径中提取有效部分
    let cleanPath;
    if (relativePath.includes('urdf/')) {
        // 提取"urdf/"及其后面的所有内容
        cleanPath = 'javascript/' + relativePath.substring(relativePath.indexOf('urdf/'));
    } else {
        // 移除开头的相对路径符号，保留javascript/前缀
        cleanPath = relativePath.replace(/^(\.\.\/)+/, 'javascript/');
    }
    
    // 构建绝对路径
    const absolutePath = window.location.origin + '/' + cleanPath;
    
    return absolutePath;
}



/**
 * 应用URDF材质和状态
 * @param {Object} robot - URDF机器人对象
 * @param {string} renderType - 渲染类型
 * @param {string} state - 状态
 * @param {Object} viewer - 查看器对象
 */
export function applyURDFMaterialAndState(robot, renderType, state, viewer) {
    // 将函数设置为全局变量，以便在其他文件中访问
    window.applyURDFMaterialAndState = applyURDFMaterialAndState;
    
    try {
        // 检查参数
        if (!robot) {
            throw new Error('Robot is null or undefined');
        }
        
        if (!viewer) {
            throw new Error('Viewer is null or undefined');
        }

        // 设置关节位置
        if (state === 'open' || state === 'close') {
            Object.values(robot.joints).forEach(joint => {
                if (joint.jointType === 'prismatic' || joint.jointType === 'revolute') {
                    const newValue = state === 'open' ? joint.limit.upper : joint.limit.lower;
                    joint.setJointValue(newValue);
                }
            });
            robot.updateMatrixWorld(true);
        }

        // 设置 mesh 材质
        robot.traverse(child => {
            if (child.isMesh) {
                if (!child.userData.originalMaterial) {
                    child.userData.originalMaterial = child.material;
                }

                switch (renderType) {
                    case 'normal':
                        child.material = normalMaterial;
                        break;
                    case 'depth':
                        try {
                            if (!viewer.camera) {
                                throw new Error('Camera not found in viewer');
                            }

                            // 计算深度范围
                            const depthRange = computeSceneDepthRange(robot.parent, viewer.camera);
                            
                            // 创建新的深度材质
                            const depthMat = depthMaterial.clone();
                            
                            // 更新材质参数
                            depthMat.uniforms.cameraNear.value = viewer.camera.near;
                            depthMat.uniforms.cameraFar.value = viewer.camera.far;
                            depthMat.uniforms.sceneDepthMin.value = depthRange.min;
                            depthMat.uniforms.sceneDepthMax.value = depthRange.max;
                            
                            // 应用材质
                            child.material = depthMat;
                        } catch (error) {
                            child.material = depthMaterial.clone();
                        }
                        break;
                    case 'wireframe':
                        child.material = wireframeMaterial;
                        break;
                    case 'segmentation':
                        // 创建一个新的黑色材质，用于所有网格
                        const blackMaterial = new THREE.MeshBasicMaterial({ color: 0x000000 });
                        
                        // 默认使用黑色材质
                        child.material = blackMaterial;
                        
                        // 获取当前选中的关节
                        const selectedJoint = window.currentSelectedJoint || state;
                        
                        // 只检查selectedJoint是否存在，完全避免处理'unselect'
                        if (selectedJoint) {
                            // 检查当前网格是否属于被选中的关节
                            let parent = child.parent;
                            let belongsToSelectedJoint = false;
                            
                            while (parent) {
                                if (parent.isURDFJoint && parent.name === selectedJoint) {
                                    belongsToSelectedJoint = true;
                                    break;
                                }
                                parent = parent.parent;
                            }
                            
                            // 如果属于被选中的关节，使用白色材质
                            if (belongsToSelectedJoint) {
                                child.material = new THREE.MeshBasicMaterial({ color: 0xffffff });
                            }
                        }
                        break;

                    default:
                        child.material = child.userData.originalMaterial;
                        child.visible = true;
                        break;
                }
            }
        });

        viewer.redraw();
    } catch (error) {
        // 恢复URDF模型可见性
        if (viewer.robot) {
            viewer.robot.visible = true;
        }
        viewer.redraw();
    }
}

/**
 * 彻底清理所有渲染模式的残留
 * @param {Object} viewer - 查看器实例
 * @param {Object} robot - 机器人对象
 */
function cleanupRenderModes(viewer, robot) {
    if (!viewer || !robot) {
        return;
    }
    
    try {
        // 1. 清理所有材质和着色器
        robot.traverse(obj => {
            if (obj.isMesh) {
                // 保存原始材质引用用于恢复
                if (!obj._originalMaterial && obj.material) {
                    obj._originalMaterial = obj.material.clone();
                }
                
                // 清除所有shader相关的自定义属性
                if (obj.material) {
                    if (Array.isArray(obj.material)) {
                        obj.material.forEach(mat => {
                            // 清除所有自定义属性
                            for (const key in mat.userData) {
                                delete mat.userData[key];
                            }
                            // 确保透明度正确
                            mat.transparent = false;
                            mat.opacity = 1.0;
                            // 重置颜色
                            if (mat.color && mat._originalColor) {
                                mat.color.copy(mat._originalColor);
                            }
                        });
                    } else {
                        // 清除所有自定义属性
                        for (const key in obj.material.userData) {
                            delete obj.material.userData[key];
                        }
                        // 确保透明度正确
                        obj.material.transparent = false;
                        obj.material.opacity = 1.0;
                        // 重置颜色
                        if (obj.material.color && obj.material._originalColor) {
                            obj.material.color.copy(obj.material._originalColor);
                        }
                    }
                }
                
                // 确保可见性重置
                obj.visible = true;
            }
        });
        
        // 2. 清理场景中的箭头对象
        removeArrowObject(viewer);
        
        // 4. 清理任何辅助对象，如包围盒、轴线等
        viewer.scene.traverse(obj => {
            if (obj.name) {
                // 清理所有辅助可视化物体，但保留默认必要的对象
                if (obj.name.includes('helper') || 
                    obj.name.includes('bbox') || 
                    obj.name.includes('outline') || 
                    obj.name.includes('highlight')) {
                    
                    if (obj.parent) {
                        obj.parent.remove(obj);
                    }
                    if (obj.geometry) obj.geometry.dispose();
                    if (obj.material) {
                        if (Array.isArray(obj.material)) {
                            obj.material.forEach(m => m.dispose());
                        } else {
                            obj.material.dispose();
                        }
                    }
                }
            }
        });
        
        // 6. 重置所有关节状态
        Object.values(robot.joints).forEach(joint => {
            if (joint.jointType === 'prismatic' || joint.jointType === 'revolute') {
                // 确保关节颜色重置
                if (joint.axis) {
                    joint.axis.traverse(obj => {
                        if (obj.isMesh && obj.material) {
                            if (obj._originalMaterial) {
                                obj.material = obj._originalMaterial.clone();
                            }
                        }
                    });
                }
            }
        });
        
        // 7. 重置背景
        viewer.scene.background = new THREE.Color(0x263238);
        
        // 8. 强制场景更新
        viewer.redraw();
    } catch (error) {
        // 错误处理
    }
}

export function updateRenderMode(mode, viewer, urdfPath, currentRenderMode, defaultLight) {
    try {
        if (!viewer || !viewer.scene || !viewer.robot) {
            return;
        }
        
        // 先执行深度清理，确保没有残留
        cleanupRenderModes(viewer, viewer.robot);
        
        // 保存新的渲染模式
        window.currentRenderMode = mode;
        
        // 为body添加当前渲染模式属性，方便CSS选择器使用
        document.body.dataset.renderMode = mode.split('-')[0];
        
        // 解析渲染模式和状态
        let renderType, state, jointId;
        if (mode.includes('-')) {
            const parts = mode.split('-');
            // 处理arrow模式，可能包含关节ID
            if (parts[0] === 'arrow') {
                renderType = `arrow-${parts[1]}`;
                if (parts.length > 2) {
                    jointId = parts[2];
                    state = parts.length > 3 ? parts[3] : 'close';
                } else {
                    jointId = window.currentJointId || "0";
                    state = 'close';
                }
            }
            else {
                renderType = parts[0];
                state = parts.slice(1).join('-');
            }
        } else {
            renderType = mode;
            state = 'original';
        }
        

        
        // 设置背景与光照
        if (renderType === 'depth' || renderType === 'wireframe' || 
            renderType === 'segmentation') {
            document.body.style.backgroundColor = '#000000';
            defaultLight.visible = false;
        } else {
            document.body.style.backgroundColor = '#263238';
            defaultLight.visible = true;
        }

        // 处理箭头模式
        if (renderType.startsWith('arrow')) {
            
            // 如果提供了关节ID，更新全局变量
            if (jointId) {
                window.currentJointId = jointId;
            }
            
            // 清除任何现有的高亮
            if (typeof clearHighlights === 'function') {
                clearHighlights();
            }
            
            loadArrowModel(urdfPath, viewer, () => {
                applyArrowMaterial(renderType);
                const stateElement = document.getElementById(`${renderType}-state`);
                const currentState = stateElement ? stateElement.value : 'close';
                
                // 应用状态到 URDF 和箭头
                if (viewer.robot) {
                    applyURDFMaterialAndState(viewer.robot, 'default', currentState, viewer);
                }
                handleArrowStateChange(viewer, currentState);
                viewer.redraw();
            });
            
            return;
        }

        // 标准URDF渲染模式
        applyURDFMaterialAndState(viewer.robot, renderType, state, viewer);
        viewer.redraw();
        
    } catch (error) {
        // 确保URDF模型可见
        if (viewer && viewer.robot) {
            viewer.robot.visible = true;
        }
        if (viewer) {
            viewer.redraw();
        }
    }
}


export let uniqueUrdfMaterial = null;