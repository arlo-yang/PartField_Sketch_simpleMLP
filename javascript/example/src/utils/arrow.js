import * as THREE from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { applyURDFMaterialAndState } from './visualManager.js';
import { initialModels } from '../models-config.js';

let arrowObject = null;
let arrowContainer = null;

/**
 * 获取当前箭头关节ID
 * @returns {string} 当前箭头关节ID
 */
export function getCurrentJointId() {
    if (typeof window.currentJointId === 'undefined') {
        window.currentJointId = "0";  // 默认使用0作为初始关节ID
    }
    return window.currentJointId;
}

/**
 * 初始化箭头容器
 * @param {Object} viewer URDF查看器对象
 */
export function initArrowContainer(viewer) {
    if (!arrowContainer) {
        arrowContainer = new THREE.Group();
        arrowContainer.name = 'arrow-container';
        viewer.scene.add(arrowContainer);
    }
}

/**
 * 加载箭头模型
 * @param {string} currentUrdfPath 当前URDF路径  
 * @param {Object} viewer URDF查看器对象
 * @param {Function} callback 加载完成后的回调函数
 */
export function loadArrowModel(currentUrdfPath, viewer, callback) {
    // 清理之前的箭头并确保容器存在
    removeArrowObject(viewer);
    initArrowContainer(viewer);
    
    const finalUrdfPath = currentUrdfPath || viewer.urdf;
    if (!finalUrdfPath) {
        if (callback) callback();
        return;
    }
    
    const pathParts = finalUrdfPath.split('/');
    const objectId = pathParts[pathParts.length - 2];
    
    // 获取当前选择的关节ID
    const jointId = getCurrentJointId();
    console.log(`Loading arrow model for joint ID: ${jointId}`);
    
    // 查找当前模型
    const model = initialModels.find(m => m.urdf.includes(`/${objectId}/`));
    if (!model) {
        console.error(`Model not found for path: ${finalUrdfPath}`);
        if (callback) callback();
        return;
    }
    
    // 获取指定关节ID的箭头OBJ
    if (model.arrowObjs && model.availableJointIds) {
        // 打印可用的关节ID，帮助调试
        console.log(`Available joint IDs for model: ${model.availableJointIds.join(', ')}`);
        console.log(`Requested joint ID: ${jointId}`);
        
        // 尝试直接使用关节ID的箭头路径
        let arrowPath;
        
        // 如果当前选择的关节ID在可用列表中，使用它
        if (model.arrowObjs[jointId]) {
            arrowPath = model.arrowObjs[jointId];
            console.log(`Found arrow model path for joint ${jointId}: ${arrowPath}`);
            loadWithFallback(arrowPath, objectId, viewer, jointId, callback);
        } 
        // 尝试使用不同格式的关节ID (如果ID是数字，尝试加上"joint_"前缀)
        else if (jointId.match(/^\d+$/) && model.arrowObjs[`joint_${jointId}`]) {
            const formattedJointId = `joint_${jointId}`;
            arrowPath = model.arrowObjs[formattedJointId];
            console.log(`Found arrow model path for formatted joint ${formattedJointId}: ${arrowPath}`);
            loadWithFallback(arrowPath, objectId, viewer, jointId, callback);
        }
        // 尝试使用不带前缀的关节ID (如果ID格式为"joint_X"，尝试只使用"X")
        else if (jointId.startsWith('joint_') && model.arrowObjs[jointId.substring(6)]) {
            const strippedJointId = jointId.substring(6);
            arrowPath = model.arrowObjs[strippedJointId];
            console.log(`Found arrow model path for stripped joint ${strippedJointId}: ${arrowPath}`);
            loadWithFallback(arrowPath, objectId, viewer, jointId, callback);
        }
        // 如果找不到匹配的关节ID，尝试构造一个可能的路径
        else {
            console.warn(`No arrow model found for joint ${jointId}, trying fallback paths`);
            // 构造一个可能的路径，基于关节ID
            const possiblePath = `/urdf/${objectId}/yy_object/yy_merged_${jointId}.obj`;
            console.log(`Trying constructed path: ${possiblePath}`);
            loadWithFallback(possiblePath, objectId, viewer, jointId, callback);
        }
    } else {
        console.warn(`No arrow objects or available joint IDs for model`);
        if (callback) callback();
    }
}

/**
 * 使用备用路径机制加载箭头模型
 * @param {string} primaryPath 主要路径
 * @param {string} objectId 对象ID
 * @param {Object} viewer 查看器对象
 * @param {string} jointId 关节ID
 * @param {Function} callback 回调函数
 */
function loadWithFallback(primaryPath, objectId, viewer, jointId, callback) {
    const loader = new OBJLoader();
    
    // 使用传入的jointId，而不是重新获取
    console.log(`Loading with fallback for joint ID: ${jointId}`);
    
    const paths = [
        primaryPath,
        `/urdf/${objectId}/yy_object/yy_merged_${jointId}.obj`,
        `./urdf/${objectId}/yy_object/yy_merged_${jointId}.obj`
    ];
    
    // 如果jointId以"joint_"开头，也尝试使用不带前缀的ID
    if (jointId.startsWith('joint_')) {
        const numericId = jointId.substring(6);
        paths.push(`/urdf/${objectId}/yy_object/yy_merged_${numericId}.obj`);
        paths.push(`./urdf/${objectId}/yy_object/yy_merged_${numericId}.obj`);
    } 
    // 如果jointId是纯数字，也尝试使用带"joint_"前缀的ID
    else if (jointId.match(/^\d+$/)) {
        paths.push(`/urdf/${objectId}/yy_object/yy_merged_joint_${jointId}.obj`);
        paths.push(`./urdf/${objectId}/yy_object/yy_merged_joint_${jointId}.obj`);
    }
    
    console.log(`Trying paths:`, paths);
    
    function tryNextPath(index = 0) {
        if (index >= paths.length) {
            console.warn(`All paths failed for joint ${jointId}`);
            if (callback) callback();
            return;
        }
        
        const currentPath = paths[index];
        console.log(`Trying path ${index + 1}/${paths.length}: ${currentPath}`);
        
        loader.load(
            currentPath,
            (object) => {
                console.log(`Successfully loaded model from ${currentPath}`);
                processLoadedModel(object, viewer);
                if (callback) callback();
            },
            (xhr) => {
                // Progress callback - can be silent
                const percentComplete = xhr.loaded / xhr.total * 100;
                if (percentComplete === 100) {
                    console.log(`Loading complete for ${currentPath}`);
                }
            },
            () => {
                // Error callback - try next path
                console.warn(`Failed to load from ${currentPath}, trying next path`);
                tryNextPath(index + 1);
            }
        );
    }
    
    // 开始尝试第一个路径
    tryNextPath();
}

/**
 * 处理加载的模型
 * @param {Object} object 加载的3D对象
 * @param {Object} viewer 查看器对象
 */
function processLoadedModel(object, viewer) {
    const arrowGroup = new THREE.Group();
    const baseGroup = new THREE.Group();
    
    object.traverse(child => {
        if (child.isMesh) {
            if (Array.isArray(child.material)) {
                const arrowGeometry = new THREE.BufferGeometry();
                const baseGeometry = new THREE.BufferGeometry();
                
                // 克隆几何属性
                Object.keys(child.geometry.attributes).forEach(key => {
                    arrowGeometry.setAttribute(key, child.geometry.attributes[key].clone());
                    baseGeometry.setAttribute(key, child.geometry.attributes[key].clone());
                });
                
                // 创建顶点索引
                const indices = [];
                const vertexCount = child.geometry.attributes.position.count;
                for (let i = 0; i < vertexCount; i++) {
                    indices.push(i);
                }
                
                // 分离箭头和基座的顶点
                const arrowVertices = [];
                const baseVertices = [];
                
                for (let i = 0; i < indices.length; i += 3) {
                    const a = indices[i];
                    const b = indices[i + 1];
                    const c = indices[i + 2];
                    
                    const materialIndex = child.geometry.groups.find(g => 
                        i >= g.start && i < g.start + g.count
                    )?.materialIndex || 0;
                    
                    if (child.material[materialIndex].name.includes('arrow')) {
                        arrowVertices.push(a, b, c);
                    } else {
                        baseVertices.push(a, b, c);
                    }
                }
                
                // 设置索引
                arrowGeometry.setIndex(arrowVertices);
                baseGeometry.setIndex(baseVertices);
                
                // 创建网格
                const arrowMesh = createMesh(arrowGeometry, child, 'arrow_material');
                const baseMesh = createMesh(baseGeometry, child, 'base_geom');
                
                arrowGroup.add(arrowMesh);
                baseGroup.add(baseMesh);
            }
        }
    });
    
    // 创建并添加箭头对象
    arrowObject = new THREE.Group();
    arrowObject.add(arrowGroup);
    arrowObject.add(baseGroup);
    
    // 添加到容器并设置变换
    arrowContainer.add(arrowObject);
    arrowObject.rotation.y = Math.PI/2;  // 使用固定的旋转角度（"close"状态）
    arrowObject.updateMatrixWorld(true);
}

/**
 * 创建网格助手函数
 * @param {Object} geometry 几何体对象
 * @param {Object} refMesh 参考网格
 * @param {string} materialName 材质名称
 * @returns {Object} 新创建的网格
 */
function createMesh(geometry, refMesh, materialName) {
    const mesh = new THREE.Mesh(geometry);
    mesh.userData.materials = [{ name: materialName }];
    mesh.position.copy(refMesh.position);
    mesh.rotation.copy(refMesh.rotation);
    mesh.scale.copy(refMesh.scale);
    return mesh;
}

/**
 * 应用箭头材质
 * @param {string} mode 箭头渲染模式
 */
export function applyArrowMaterial(mode) {
    if (!arrowObject) return;
    
    // 清理已有的边缘线和点
    arrowObject.traverse(child => {
        if (child.isLineSegments || child.isPoints) {
            child.parent.remove(child);
        }
    });
    
    // 分别处理箭头和基座
    arrowObject.traverse(child => {
        if (child.isMesh) {
            const isArrowPart = child.userData.materials[0].name.toLowerCase().includes('arrow');
            
            switch(mode) {
                case 'arrow-sketch':
                    if (isArrowPart) {
                        // 创建素描线框效果
                        createSketchLinesForMesh(child);
                        child.material = new THREE.MeshBasicMaterial({ visible: false });
                    } else {
                        child.material = createBaseMaterial();
                    }
                    break;
                    
                case 'arrow-wireframe':
                    // 线框效果
                    child.material = new THREE.MeshBasicMaterial({
                        wireframe: true,
                        color: isArrowPart ? 0xff8c00 : 0xffffff,
                        side: THREE.DoubleSide
                    });
                    break;
                    
                default:
                    // 默认arrow-default模式
                    child.material = new THREE.MeshPhongMaterial({
                        color: isArrowPart ? 0xff8c00 : 0xffffff,
                        emissive: isArrowPart ? 0xff4500 : 0x000000,
                        emissiveIntensity: isArrowPart ? 0.5 : 0,
                        side: THREE.DoubleSide
                    });
            }
        }
    });
}

/**
 * 为网格创建素描线框效果
 * @param {Object} mesh 网格对象
 */
function createSketchLinesForMesh(mesh) {
    const edges = new THREE.EdgesGeometry(mesh.geometry, 30);
    const line = new THREE.LineSegments(
        edges,
        new THREE.LineBasicMaterial({ 
            color: 0xff8c00,
            linewidth: 2
        })
    );
    line.position.copy(mesh.position);
    line.rotation.copy(mesh.rotation);
    line.scale.copy(mesh.scale);
    mesh.parent.add(line);
}

/**
 * 创建基座材质
 * @returns {Object} 基座材质
 */
function createBaseMaterial() {
    return new THREE.MeshPhongMaterial({
        color: 0xffffff,
        side: THREE.DoubleSide
    });
}

/**
 * 移除箭头对象
 * @param {Object} viewer URDF查看器对象
 */
export function removeArrowObject(viewer) {
    if (arrowContainer) {
        while (arrowContainer.children.length > 0) {
            const child = arrowContainer.children[0];
            arrowContainer.remove(child);
            if (child.geometry) {
                child.geometry.dispose();
            }
            if (child.material) {
                if (Array.isArray(child.material)) {
                    child.material.forEach(m => m.dispose());
                } else {
                    child.material.dispose();
                }
            }
        }
    }
    arrowObject = null;
}

/**
 * 移除箭头容器
 * @param {Object} viewer URDF查看器对象
 */
export function removeArrowContainer(viewer) {
    if (arrowContainer) {
        removeArrowObject(viewer);
        viewer.scene.remove(arrowContainer);
        arrowContainer = null;
    }
}

/**
 * 处理箭头状态变化
 * @param {Object} viewer URDF查看器对象
 * @param {string} state 状态：'open'或'close'
 */
export function handleArrowStateChange(viewer, state) {
    if (!arrowObject) {
        return;
    }
    
    // 根据状态设置箭头旋转
    arrowObject.rotation.y = state === 'open' ? Math.PI : Math.PI/2;
    
    // 更新世界矩阵
    arrowObject.updateMatrixWorld(true);
    
    // 重绘场景
    if (viewer) {
        viewer.redraw();
    }
}

/**
 * 处理箭头关节ID变化
 * @param {Object} viewer URDF查看器对象
 * @param {string} currentUrdfPath 当前URDF路径
 * @param {string} currentRenderMode 当前渲染模式
 * @param {string} arrowMode 箭头渲染模式
 * @param {string} jointId 箭头关节ID
 * @returns {Promise} 处理完成的Promise
 */
export function handleArrowJointIdChange(viewer, currentUrdfPath, currentRenderMode, arrowMode, jointId) {
    console.group(`Changing arrow joint to: ${jointId}`);
    
    // 更新当前箭头关节ID
    if (jointId !== undefined) {
        console.log(`Setting current joint ID from ${window.currentJointId} to ${jointId}`);
        window.currentJointId = jointId;
    } else {
        console.warn('No joint ID provided, using current:', window.currentJointId);
    }
    
    // 返回Promise以便于异步等待
    return new Promise((resolve) => {
        console.log(`Loading arrow model for joint ${jointId} with mode ${arrowMode}`);
    
    // 重新加载带有新关节ID的箭头模型
    loadArrowModel(currentUrdfPath, viewer, () => {
        // 应用箭头材质并重绘
            console.log(`Applying ${arrowMode} material to arrow model`);
        applyArrowMaterial(arrowMode);
        viewer.redraw();
            
            console.log(`Arrow joint change complete for joint ${jointId}`);
            console.groupEnd();
            
            // 使用短暂延迟确保渲染完成
            setTimeout(resolve, 100);
        });
    });
}

/**
 * 获取当前模型可用的箭头关节ID
 * @param {string} urdfPath URDF路径
 * @returns {Array} 可用的箭头关节ID数组
 */
export function getAvailableJointIds(urdfPath) {
    if (!urdfPath) return ["0"];
    
    const pathParts = urdfPath.split('/');
    const objectId = pathParts[pathParts.length - 2];
    
    // 查找当前模型
    const model = initialModels.find(m => m.urdf.includes(`/${objectId}/`));
    
    // 返回可用的关节ID
    if (model && model.availableJointIds && model.availableJointIds.length > 0) {
        return model.availableJointIds;
    }
    
    // 如果没有找到任何关节ID，则返回默认["0"]
    return ["0"];
} 