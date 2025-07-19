import * as THREE from 'three';
import { depthMaterial } from './visualManager.js';
import { getCurrentModelInfo } from './modelManager.js';
import { updateRenderMode } from './visualManager.js';
import { uncheckAllRenders } from './uiControls.js';
import { setColor } from './visualManager.js';
import { loadArrowModel, applyArrowMaterial, removeArrowObject, handleArrowJointIdChange } from './arrow.js';

// 保存截图时的渲染状态
/**
 * 捕获并保存截图
 * 截图不会应用放大效果，保持原始比例
 * @param {Object} viewer - 3D查看器对象
 * @param {string} viewLabel - 视角标签，用于文件命名
 * @param {boolean} downloadJson - 是否下载JSON参数文件
 * @param {boolean} validate - 是否显示验证黑框
 * @param {boolean} isValidation - 是否是验证模式（内部使用）
 * @param {string} renderMode - 渲染模式
 * @returns {Promise<Object|null>} 截图数据和相机参数，或null（如果出错）
 */
async function captureScreenshot(viewer, viewLabel = '', downloadJson = true, validate = true, isValidation = false, renderMode = 'segmentation') {
    console.group('Capturing Screenshot');
    
    // 获取当前模型信息
    const modelInfo = getCurrentModelInfo();
    if (!modelInfo) {
        console.error('No model info available');
        console.groupEnd();
        return;
    }

    // 获取类别名称 (首字母大写)
    const category = modelInfo.type.charAt(0).toUpperCase() + modelInfo.type.slice(1);
    
    console.log('Screenshot info:', {
        category: category,
        modelId: modelInfo.id,
        viewLabel: viewLabel,
        renderMode: renderMode
    });

    const currentRenderMode = getCurrentRenderMode() || renderMode;
    const needsBlackBackground = currentRenderMode === 'depth' || currentRenderMode === 'segmentation' || currentRenderMode === 'wireframe';
    const isArrowSketchMode = currentRenderMode === 'arrow-sketch';

    // 保存原始背景色
    const originalBackground = viewer.scene.background;

    // 设置背景色 - 特殊模式使用黑色背景，其他模式使用透明背景
    if (needsBlackBackground) {
        viewer.scene.background = new THREE.Color(0x000000);
    } else {
        // 设置透明背景
        viewer.scene.background = null;
    }

    // 保存原始状态
    const originalSize = {
        width: viewer.renderer.domElement.width,
        height: viewer.renderer.domElement.height,
        aspect: viewer.camera.aspect
    };
    
    // 保存原始相机位置和FOV
    const originalPosition = viewer.camera.position.clone();
    const originalFOV = viewer.camera.fov;
    
    try {
        // 确保场景已经完全加载
        if (!viewer.robot) {
            console.error('Robot not loaded');
            return;
        }

        // 创建一个512x512的临时渲染目标
        const renderTarget = new THREE.WebGLRenderTarget(512, 512, {
            minFilter: THREE.LinearFilter,
            magFilter: THREE.LinearFilter,
            format: THREE.RGBAFormat
        });

        // 计算保持宽高比的渲染尺寸
        const targetAspect = 1; // 512/512
        const currentAspect = viewer.camera.aspect;
        let renderWidth, renderHeight;
        
        if (currentAspect > targetAspect) {
            // 当前画面更宽
            renderHeight = 512;
            renderWidth = Math.round(512 * currentAspect);
        } else {
            // 当前画面更高
            renderWidth = 512;
            renderHeight = Math.round(512 / currentAspect);
        }

        // 设置临时渲染尺寸
        viewer.renderer.setSize(renderWidth, renderHeight, false);
        
        // 不应用放大效果，保持原始比例
        const zoomFactor = 1.0; // 100% 不放大
        
        // 计算从目标点到相机的方向向量
        const target = viewer.controls ? new THREE.Vector3(
            viewer.controls.target.x,
            viewer.controls.target.y,
            viewer.controls.target.z
        ) : new THREE.Vector3(0, 0, 0);
        
        const directionToTarget = new THREE.Vector3().subVectors(target, originalPosition);
        directionToTarget.normalize();
        
        // 计算新的相机位置：沿着相机到目标的方向靠近目标
        // 由于zoomFactor为1.0，相机位置不会改变
        const distanceToTarget = originalPosition.distanceTo(target);
        const newDistance = distanceToTarget / zoomFactor;
        const positionOffset = directionToTarget.multiplyScalar(distanceToTarget - newDistance);
        
        // 移动相机位置
        viewer.camera.position.copy(originalPosition).add(positionOffset);
        
        // 更新相机
        viewer.camera.updateProjectionMatrix();

        // 强制更新场景
        viewer.redraw();
        
        // 等待一帧以确保渲染完成
        requestAnimationFrame(async () => {
            try {
                // 创建一个512x512的canvas
                const canvas = document.createElement('canvas');
                canvas.width = 512;
                canvas.height = 512;
                const ctx = canvas.getContext('2d');

                // 如果是需要黑色背景的模式，先填充黑色背景
                if (needsBlackBackground) {
                    ctx.fillStyle = '#000000';
                    ctx.fillRect(0, 0, 512, 512);
                } else {
                    // 不填充背景，保持透明
                    ctx.clearRect(0, 0, 512, 512);
                }

                // 计算居中位置
                const x = Math.round((512 - renderWidth) / 2);
                const y = Math.round((512 - renderHeight) / 2);

                // 捕获渲染器内容并绘制到canvas上
                const rendererCanvas = viewer.renderer.domElement;
                ctx.drawImage(rendererCanvas, x, y, renderWidth, renderHeight);

                // 获取最终图像数据 - 确保PNG格式支持透明度
                const imgData = canvas.toDataURL('image/png');
                
                // 准备相机参数 - 使用原始相机参数，不是放大后的
                const cameraParams = {
                    fov: originalFOV,  // 使用原始FOV
                    aspect: viewer.camera.aspect,
                    near: viewer.camera.near,
                    far: viewer.camera.far,
                    position: {
                        x: originalPosition.x,  // 使用原始位置
                        y: originalPosition.y,
                        z: originalPosition.z
                    },
                    rotation: {
                        x: viewer.camera.rotation.x,
                        y: viewer.camera.rotation.y,
                        z: viewer.camera.rotation.z
                    },
                    quaternion: {
                        x: viewer.camera.quaternion.x,
                        y: viewer.camera.quaternion.y,
                        z: viewer.camera.quaternion.z,
                        w: viewer.camera.quaternion.w
                    },
                    up: viewer.camera.up.toArray(),
                    target: viewer.controls ? {
                        x: viewer.controls.target.x,
                        y: viewer.controls.target.y,
                        z: viewer.controls.target.z
                    } : null,
                    viewInfo: {
                        name: viewLabel,
                        type: viewer.camera.isOrthographicCamera ? 'orthographic' : 'perspective',
                        up: Array.from(viewer.up),
                        target: viewer.controls ? Array.from(viewer.controls.target) : [0, 0, 0]
                    },
                    projectionMatrix: Array.from(viewer.camera.projectionMatrix.elements),
                    viewMatrix: Array.from(viewer.camera.matrixWorldInverse.elements),
                    matrixWorld: Array.from(viewer.camera.matrixWorld.elements),
                    imageWidth: 512,
                    imageHeight: 512,
                    renderWidth: renderWidth,
                    renderHeight: renderHeight,
                    offsetX: x,
                    offsetY: y,

                    // 添加新的信息
                    modelInfo: {
                        id: modelInfo.id,
                        scale: Array.from(viewer.robot.scale),
                        position: Array.from(viewer.robot.position),
                        rotation: Array.from(viewer.robot.rotation),
                        boundingBox: viewer.robot?.geometry?.boundingBox ? {
                            min: Array.from(viewer.robot.geometry.boundingBox.min),
                            max: Array.from(viewer.robot.geometry.boundingBox.max)
                        } : null
                    },

                    renderSettings: {
                        pixelRatio: window.devicePixelRatio,
                        antialias: viewer.renderer.antialias,
                        encoding: viewer.renderer.outputEncoding,
                        gammaFactor: viewer.renderer.gammaFactor,
                        shadowMap: viewer.renderer.shadowMap.enabled
                    },

                    sceneInfo: {
                        backgroundColor: viewer.scene.background ? '#' + viewer.scene.background.getHexString() : null,
                        fogEnabled: !!viewer.scene.fog,
                        lightingSetup: viewer.scene.children
                            .filter(child => child.isLight)
                            .map(light => ({
                                type: light.type,
                                position: Array.from(light.position),
                                intensity: light.intensity,
                                color: '#' + light.color.getHexString()
                            }))
                    },

                    timestamp: Date.now(),

                    depthInfo: {
                        depthNear: viewer.camera.near,
                        depthFar: viewer.camera.far,
                        linearDepth: true,
                        depthPacking: 'BasicDepthPacking'
                    }
                };

                // 获取当前信息
                const modelId = modelInfo.id;
                
                // 修改文件名生成部分
                const baseFileName = viewLabel ? 
                    `${category}_${modelId}_${renderMode}_${viewLabel}` :
                    `${category}_${modelId}_${renderMode}_${getCurrentViewMode()}`;
                
                // 下载 JSON
                const jsonData = JSON.stringify(cameraParams, null, 2);
                const jsonBlob = new Blob([jsonData], { type: 'application/json' });
                const jsonUrl = URL.createObjectURL(jsonBlob);
                
                // 如果是arrow-sketch模式，获取并导出箭头线段的2D坐标
                let arrowLinesData = null;
                if (isArrowSketchMode) {
                    const arrowLines2D = getArrowLines2DCoordinates(viewer, x, y, renderWidth, renderHeight);
                    if (arrowLines2D && arrowLines2D.length > 0) {
                        // 创建包含2D坐标的文本
                        let linesText = "# Arrow Lines 2D Coordinates (x, y)\n";
                        linesText += "# Format: line_index, point_index, 3D_x, 3D_y, 3D_z, 2D_x, 2D_y\n";
                        arrowLines2D.forEach((line, lineIndex) => {
                            line.points.forEach((point, pointIndex) => {
                                linesText += `${lineIndex}, ${pointIndex}, ${point.position3D.x.toFixed(6)}, ${point.position3D.y.toFixed(6)}, ${point.position3D.z.toFixed(6)}, ${point.position2D.x.toFixed(2)}, ${point.position2D.y.toFixed(2)}\n`;
                            });
                        });
                        
                        // 创建文本文件Blob
                        arrowLinesData = new Blob([linesText], { type: 'text/plain' });
                    }
                }
                
                // 下载文件
                if (downloadJson) {
                    // 准备下载任务
                    const downloadTasks = [
                        // 下载PNG
                        new Promise(r => {
                            const imgLink = document.createElement('a');
                            imgLink.download = `${baseFileName}.png`;
                            imgLink.href = imgData;
                            document.body.appendChild(imgLink);
                            imgLink.click();
                            document.body.removeChild(imgLink);
                            setTimeout(r, 100);
                        }),
                        // 下载JSON
                        new Promise(r => {
                            const jsonLink = document.createElement('a');
                            jsonLink.download = `${baseFileName}_params.json`;
                            jsonLink.href = jsonUrl;
                            document.body.appendChild(jsonLink);
                            jsonLink.click();
                            document.body.removeChild(jsonLink);
                            URL.revokeObjectURL(jsonUrl);
                            setTimeout(r, 100);
                        })
                    ];
                    
                    // 如果是arrow-sketch模式，也下载线段坐标
                    if (isArrowSketchMode && arrowLinesData) {
                        downloadTasks.push(
                            new Promise(r => {
                                const linesUrl = URL.createObjectURL(arrowLinesData);
                                const linesLink = document.createElement('a');
                                linesLink.download = `${baseFileName}_lines.txt`;
                                linesLink.href = linesUrl;
                                document.body.appendChild(linesLink);
                                linesLink.click();
                                document.body.removeChild(linesLink);
                                URL.revokeObjectURL(linesUrl);
                                setTimeout(r, 100);
                            })
                        );
                    }
                    
                    // 执行所有下载任务
                    await Promise.all(downloadTasks);
                } else {
                    // 只下载 PNG
                    await new Promise(r => {
                        const imgLink = document.createElement('a');
                        imgLink.download = `${baseFileName}.png`;
                        imgLink.href = imgData;
                        document.body.appendChild(imgLink);
                        imgLink.click();
                        document.body.removeChild(imgLink);
                        setTimeout(r, 100);
                    });
                    
                    // 如果是arrow-sketch模式，也下载线段坐标
                    if (isArrowSketchMode && arrowLinesData) {
                        await new Promise(r => {
                            const linesUrl = URL.createObjectURL(arrowLinesData);
                            const linesLink = document.createElement('a');
                            linesLink.download = `${baseFileName}_lines.txt`;
                            linesLink.href = linesUrl;
                            document.body.appendChild(linesLink);
                            linesLink.click();
                            document.body.removeChild(linesLink);
                            URL.revokeObjectURL(linesUrl);
                            setTimeout(r, 100);
                        });
                    }
                }

                // 恢复原始背景色
                viewer.scene.background = originalBackground;

                // 在下载文件后添加验证
                if (validate && !isValidation) {
                    await validateCameraParams(viewer, imgData, cameraParams);
                }

                console.log('Screenshot captured successfully');
                return { imgData, cameraParams };
            } catch (error) {
                console.error('Error capturing screenshot:', error);
                return null;
            } finally {
                // 恢复相机原始位置
                viewer.camera.position.copy(originalPosition);
                viewer.camera.fov = originalFOV;
                viewer.camera.updateProjectionMatrix();
                
                // 恢复原始渲染尺寸
                viewer.renderer.setSize(originalSize.width, originalSize.height, false);
                viewer.camera.aspect = originalSize.aspect;
                viewer.camera.updateProjectionMatrix();
                
                // 恢复原始背景色
                viewer.scene.background = originalBackground;
                
                // 强制更新场景
                viewer.redraw();
            }
        });
    } catch (error) {
        console.error('Error in captureScreenshot:', error);
        
        // 恢复相机原始位置
        viewer.camera.position.copy(originalPosition);
        viewer.camera.fov = originalFOV;
        viewer.camera.updateProjectionMatrix();
        
        // 恢复原始渲染尺寸
        viewer.renderer.setSize(originalSize.width, originalSize.height, false);
        viewer.camera.aspect = originalSize.aspect;
        viewer.camera.updateProjectionMatrix();
        
        // 恢复原始背景色
        viewer.scene.background = originalBackground;
        
        // 强制更新场景
        viewer.redraw();
    }
    
    console.groupEnd();
}

/**
 * 获取箭头线段的3D坐标并转换为2D屏幕坐标
 * @param {Object} viewer - 3D查看器对象
 * @param {number} offsetX - 渲染区域在canvas中的X偏移
 * @param {number} offsetY - 渲染区域在canvas中的Y偏移
 * @param {number} renderWidth - 渲染区域宽度
 * @param {number} renderHeight - 渲染区域高度
 * @returns {Array} 包含线段及其点的3D和2D坐标的数组
 */
function getArrowLines2DCoordinates(viewer, offsetX, offsetY, renderWidth, renderHeight) {
    console.group('Getting Arrow Line 2D Coordinates');
    
    try {
        const lines = [];
        
        // 查找箭头容器
        const arrowContainer = viewer.scene.children.find(child => child.name === 'arrow-container');
        if (!arrowContainer) {
            console.warn('Arrow container not found');
            console.groupEnd();
            return [];
        }
        
        // 遍历场景中的所有LineSegments对象
        arrowContainer.traverse(child => {
            if (child instanceof THREE.LineSegments) {
                console.log('Found LineSegments object:', child.name || 'unnamed');
                
                // 获取线段的位置属性
                const positions = child.geometry.attributes.position;
                const posArray = positions.array;
                
                // 创建世界坐标矩阵
                child.updateMatrixWorld(true);
                const worldMatrix = child.matrixWorld;
                
                // 创建投影矩阵
                const projScreenMatrix = new THREE.Matrix4();
                projScreenMatrix.multiplyMatrices(
                    viewer.camera.projectionMatrix,
                    viewer.camera.matrixWorldInverse
                );
                
                // 线段数据结构
                const lineData = {
                    points: []
                };
                
                // 遍历所有点（每两个点构成一条线段）
                for (let i = 0; i < posArray.length; i += 3) {
                    // 创建局部坐标向量
                    const localPos = new THREE.Vector3(
                        posArray[i],
                        posArray[i + 1],
                        posArray[i + 2]
                    );
                    
                    // 转换为世界坐标
                    const worldPos = localPos.clone().applyMatrix4(worldMatrix);
                    
                    // 转换为标准化设备坐标 (NDC)
                    const ndcPos = worldPos.clone().project(viewer.camera);
                    
                    // 转换为屏幕坐标 (像素)
                    const screenX = Math.round(((ndcPos.x + 1) / 2) * renderWidth + offsetX);
                    const screenY = Math.round(((-ndcPos.y + 1) / 2) * renderHeight + offsetY);
                    
                    // 添加到线段点数组
                    lineData.points.push({
                        position3D: worldPos,
                        position2D: { x: screenX, y: screenY }
                    });
                }
                
                // 添加线段数据到结果数组
                lines.push(lineData);
                console.log(`Processed ${lineData.points.length} line segment points`);
            }
        });
        
        console.log(`Total found ${lines.length} line segments, total ${lines.reduce((sum, line) => sum + line.points.length, 0)} points`);
        console.groupEnd();
        return lines;
    } catch (error) {
        console.error('Error getting arrow line coordinates:', error);
        console.groupEnd();
        return [];
    }
}

// 捕获深度信息
function captureDepthInfo(viewer) {
    const depthTarget = new THREE.WebGLRenderTarget(512, 512, {
        type: THREE.FloatType,
        format: THREE.RGBAFormat
    });
    
    // 保存原始材质
    const originalMaterials = new Map();
    const depthMat = depthMaterial.clone();
    depthMat.uniforms.cameraNear.value = viewer.camera.near;
    depthMat.uniforms.cameraFar.value = viewer.camera.far;

    viewer.scene.traverse(object => {
        if (object.isMesh) {
            originalMaterials.set(object, object.material);
            object.material = depthMat;
        }
    });
    
    // 渲染深度图
    viewer.renderer.setRenderTarget(depthTarget);
    viewer.renderer.render(viewer.scene, viewer.camera);
    
    // 读取深度缓冲
    const depthBuffer = new Float32Array(512 * 512 * 4);
    viewer.renderer.readRenderTargetPixels(depthTarget, 0, 0, 512, 512, depthBuffer);
    
    // 提取线性深度值
    const linearDepthBuffer = new Float32Array(512 * 512);
    for (let i = 0; i < depthBuffer.length; i += 4) {
        linearDepthBuffer[i/4] = (depthBuffer[i] + depthBuffer[i+1] + depthBuffer[i+2]) / 3;
    }
    
    // 恢复原始材质
    viewer.scene.traverse(object => {
        if (object.isMesh) {
            object.material = originalMaterials.get(object);
        }
    });
    
    // 清理资源
    viewer.renderer.setRenderTarget(null);
    depthTarget.dispose();
    
    return linearDepthBuffer;
}

// 获取当前渲染模式
function getCurrentRenderMode() {
    const renderButtons = {
        'default': document.getElementById('default-render'),
        'normal': document.getElementById('normal-render'),
        'depth': document.getElementById('depth-render'),
        'wireframe': document.getElementById('wireframe-render'),
        'segmentation': document.getElementById('segmentation-render'),
        'arrow-default': document.getElementById('arrow-default'),
        'arrow-sketch': document.getElementById('arrow-sketch'),
        'arrow-wireframe': document.getElementById('arrow-wireframe')
    };

    for (const [mode, button] of Object.entries(renderButtons)) {
        if (button?.classList.contains('checked')) return mode;
    }
    return 'default';
}

async function downloadFiles(screenshot, cameraParams, modelInfo, renderMode, viewMode) {
    try {
        // 检查视图模式是否包含关节信息（格式如 "top_left_joint1"）
        let viewModeBase = viewMode;
        let jointSuffix = '';
        
        // 提取关节后缀（如果存在）
        const jointMatch = viewMode.match(/_joint([^_]+)$/);
        if (jointMatch) {
            jointSuffix = jointMatch[0]; // 包含 "_joint{id}"
            viewModeBase = viewMode.replace(jointMatch[0], ''); // 移除关节后缀
        }
        
        // 构建文件名
        const baseFileName = `${modelInfo.id}_${modelInfo.type}_${renderMode}_${viewModeBase}${jointSuffix}`;
        const paramsFileName = `${baseFileName}_params.json`;
        const imageFileName = `${baseFileName}.png`;
        
        console.log(`Downloading file: ${imageFileName}`);
        
        // 下载JSON文件
        const paramsBlob = new Blob([JSON.stringify(cameraParams, null, 2)], { type: 'application/json' });
        const paramsUrl = URL.createObjectURL(paramsBlob);
        
        await new Promise((resolve) => {
            const link = document.createElement('a');
            link.href = paramsUrl;
            link.download = paramsFileName;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(paramsUrl);
            setTimeout(resolve, 100);
        });
        
        // 下载图片文件
        await new Promise((resolve) => {
            const link = document.createElement('a');
            link.href = screenshot;
            link.download = imageFileName;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            setTimeout(resolve, 100);
        });
    } catch (error) {
        console.error('Error during download:', error);
    }
}

function takeScreenshot(viewer, currentUrdfPath) {
    console.log('Taking screenshot with urdfPath:', currentUrdfPath);
    
    // 捕获截图
    const screenshot = captureScreenshot(viewer);
    console.log('Screenshot captured');
    
    // 捕获深度信息
    const linearDepthBuffer = captureDepthInfo(viewer);
    console.log('Depth info captured');
    
    // 准备相机参数
    const cameraParams = {
        fov: viewer.camera.fov,
        aspect: viewer.camera.aspect,
        near: viewer.camera.near,
        far: viewer.camera.far,
        position: {
            x: viewer.camera.position.x,
            y: viewer.camera.position.y,
            z: viewer.camera.position.z
        },
        rotation: {
            x: viewer.camera.rotation.x,
            y: viewer.camera.rotation.y,
            z: viewer.camera.rotation.z
        },
        quaternion: {
            x: viewer.camera.quaternion.x,
            y: viewer.camera.quaternion.y,
            z: viewer.camera.quaternion.z,
            w: viewer.camera.quaternion.w
        },
        projectionMatrix: Array.from(viewer.camera.projectionMatrix.elements),
        viewMatrix: Array.from(viewer.camera.matrixWorldInverse.elements),
        matrixWorld: Array.from(viewer.camera.matrixWorld.elements),
        imageWidth: 512,
        imageHeight: 512,
        target: {
            x: viewer.controls.target.x,
            y: viewer.controls.target.y,
            z: viewer.controls.target.z
        },
        depthInfo: {
            buffer: Array.from(linearDepthBuffer),
            near: viewer.camera.near,
            far: viewer.camera.far,
            linearizeCoefficient: {
                a: viewer.camera.far / (viewer.camera.far - viewer.camera.near),
                b: (-viewer.camera.far * viewer.camera.near) / (viewer.camera.far - viewer.camera.near)
            }
        }
    };
    
    // 获取当前状态信息
    const modelInfo = getCurrentModelInfo();
    console.log('Model info retrieved:', modelInfo);
    
    const renderMode = getCurrentRenderMode();
    console.log('Current render mode:', renderMode);
    
    const viewMode = document.getElementById('view-select').value;
    console.log('Current view mode:', viewMode);

    // 下载文件
    downloadFiles(screenshot, cameraParams, modelInfo, renderMode, viewMode);
}

function addScreenshotControl(viewer, currentUrdfPath) {
    console.group('Adding Screenshot Control');
    
    const screenshotButton = document.getElementById('screenshot-toggle');
    if (!screenshotButton) {
        console.error('Screenshot button not found');
        console.groupEnd();
        return;
    }

    // 移除旧的事件监听器
    if (screenshotButton._clickListener) {
        screenshotButton.removeEventListener('click', screenshotButton._clickListener);
    }

    // 添加新的事件监听器
    const newListener = () => {
        if (!screenshotButton.classList.contains('checked')) {
            console.log('Screenshot button clicked with urdfPath:', currentUrdfPath);
            screenshotButton.classList.add('checked');
            // 显式指定下载JSON文件(true)和显示验证黑框(true)
            captureScreenshot(viewer, '', true, true);
            setTimeout(() => screenshotButton.classList.remove('checked'), 200);
        }
    };
    
    screenshotButton._clickListener = newListener;
    screenshotButton.addEventListener('click', newListener);
    
    console.log('Screenshot control added successfully');
    console.groupEnd();
}

// Add this function to handle the gallery control
function addGalleryControl(viewer, currentUrdfPath) {
    console.group('Adding Gallery Control');
    
    const galleryButton = document.getElementById('gallery-toggle');
    if (!galleryButton) {
        console.error('Gallery button not found');
        console.groupEnd();
        return;
    }

    // 移除旧的事件监听器
    if (galleryButton._clickListener) {
        galleryButton.removeEventListener('click', galleryButton._clickListener);
    }

    // 添加新的事件监听器
    const newListener = async () => {
        if (!galleryButton.classList.contains('checked')) {
            try {
                console.log('Starting gallery capture');
                
                // 直接设置全局变量为null，避免任何unselect影响
                window.currentSelectedJoint = null;
                
                // 确保关节选择器UI重置为无选择状态
                const jointSelect = document.getElementById('segmentation-joint-select');
                if (jointSelect) {
                    jointSelect.value = 'unselect';
                }
                
                galleryButton.classList.add('checked');
                
                // 创建模式选择对话框
                const result = await new Promise(resolve => {
                    const modeDialog = document.createElement('div');
                    modeDialog.style.cssText = `
                        position: fixed;
                        top: 50%;
                        left: 50%;
                        transform: translate(-50%, -50%);
                        background: rgba(0, 0, 0, 0.9);
                        color: white;
                        padding: 20px;
                        border-radius: 10px;
                        z-index: 10000;
                        text-align: center;
                    `;
                    
                    modeDialog.innerHTML = `
                        <h3>Select Capture Mode</h3>
                        <div style="display: flex; flex-direction: column; gap: 10px; margin: 15px 0;">
                            <button id="capture-current" style="padding: 10px 20px; background: #4CAF50; border: none; border-radius: 4px; color: white; cursor: pointer;">Capture Current Model</button>
                            <button id="capture-all" style="padding: 10px 20px; background: #2196F3; border: none; border-radius: 4px; color: white; cursor: pointer;">Capture All Models</button>
                            <button id="capture-cancel" style="padding: 10px 20px; background: #f44336; border: none; border-radius: 4px; color: white; cursor: pointer; margin-top: 10px;">Cancel</button>
                        </div>
                    `;
                    
                    document.body.appendChild(modeDialog);
                    
                    document.getElementById('capture-current').onclick = () => {
                        document.body.removeChild(modeDialog);
                        resolve({ mode: 'current' });
                    };
                    
                    document.getElementById('capture-all').onclick = () => {
                        document.body.removeChild(modeDialog);
                        resolve({ mode: 'all' });
                    };
                    
                    document.getElementById('capture-cancel').onclick = () => {
                        document.body.removeChild(modeDialog);
                        resolve({ mode: 'cancel' });
                    };
                });
                
                if (result.mode === 'current') {
                    await captureGallery(viewer);
                } else if (result.mode === 'all') {
                    await captureAllModels(viewer);
                }
                // 如果是cancel，不执行任何操作
            } catch (error) {
                console.error('Gallery capture failed:', error);
            } finally {
                galleryButton.classList.remove('checked');
            }
        }
    };
    
    galleryButton._clickListener = newListener;
    galleryButton.addEventListener('click', newListener);
    
    console.log('Gallery control added successfully');
    console.groupEnd();
}

// 其他辅助函数
function getCurrentModelId(viewer) {
    const currentModel = document.querySelector('.model-item[urdf="' + window.currentUrdfPath + '"]');
    if (!currentModel) return 'unknown';
    return currentModel.textContent.trim(); // 确保移除任何空白字符
}

export function getCurrentViewMode() {
    const viewSelect = document.getElementById('view-select');
    return viewSelect ? viewSelect.value : '';
}

/**
 * 捕获所有视角的多种渲染模式图像
 * 该函数可以针对当前模型或所有模型执行捕获
 * @param {Object} viewer - 3D查看器对象
 * @param {boolean} captureAllModels - 是否捕获所有模型（而不仅是当前模型）
 * @returns {Promise<void>}
 */
async function captureGallery(viewer) {
    console.group('Capturing Gallery');
    
    // 获取当前模型信息
    const modelInfo = getCurrentModelInfo();
    if (!modelInfo) {
        console.error('Cannot get current model information');
        console.groupEnd();
        return;
    }

    // 保存当前视角和渲染模式
    const originalView = {
        position: viewer.camera.position.clone(),
        target: viewer.controls.target.clone()
    };
    const originalRenderMode = getCurrentRenderMode();
    const originalBackground = viewer.scene.background;

    // 创建进度指示器
    const progressContainer = document.createElement('div');
    progressContainer.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: rgba(0, 0, 0, 0.8);
        color: white;
        padding: 20px;
        border-radius: 10px;
        z-index: 9999;
        text-align: center;
        min-width: 300px;
    `;
    
    const progressTitle = document.createElement('h3');
    progressTitle.textContent = 'Multi-Mode Image Capture Progress';
    progressTitle.style.margin = '0 0 15px 0';
    
    const progressText = document.createElement('div');
    progressText.style.marginBottom = '10px';
    progressText.textContent = 'Preparing...';
    
    const progressBarContainer = document.createElement('div');
    progressBarContainer.style.cssText = `
        width: 100%;
        background-color: #333;
        border-radius: 5px;
        margin-bottom: 10px;
        overflow: hidden;
    `;
    
    const progressBar = document.createElement('div');
    progressBar.style.cssText = `
        height: 10px;
        background-color: #4CAF50;
        width: 0%;
        transition: width 0.3s;
    `;
    
    progressBarContainer.appendChild(progressBar);
    progressContainer.appendChild(progressTitle);
    progressContainer.appendChild(progressText);
    progressContainer.appendChild(progressBarContainer);
    
    document.body.appendChild(progressContainer);

    try {
        // 定义多种渲染模式 - 修改顺序为 default, normal, depth, segmentation, arrow-sketch
        const renderModes = [
            { 
                mode: 'default', 
                needsJoints: false, 
                blackBg: false,
                buttonId: 'default-render'
            },
            { 
                mode: 'normal', 
                needsJoints: false, 
                blackBg: false,
                buttonId: 'normal-render'
            },
            { 
                mode: 'depth', 
                needsJoints: false, 
                blackBg: true,
                buttonId: 'depth-render'
            },
            { 
                mode: 'segmentation', 
                needsJoints: true, 
                blackBg: true,
                buttonId: 'segmentation-render'
            },
            { 
                mode: 'arrow-sketch', 
                needsJoints: true,  // 修改为true，使用与segmentation相同的关节处理
                blackBg: false,     // 确保使用透明背景
                buttonId: 'arrow-sketch'
            }
        ];
        
        // 获取视角列表
        const viewNames = Object.keys(window.viewPositions);
        
        // 获取可动关节列表
        const jointSelect = document.getElementById('segmentation-joint-select');
        let jointOptions = [];
        
        if (jointSelect) {
            jointOptions = Array.from(jointSelect.options)
                .map(option => option.value)
                .filter(value => value !== 'unselect');
        }
        
        console.log(`Found ${viewNames.length} views and ${jointOptions.length} joints`);
        
        // 计算总任务数
        let totalTasks = 0;
        for (const mode of renderModes) {
            // 更新总任务数计算
            if (mode.mode === 'arrow-sketch') {
                // 对于arrow-sketch模式，单独处理任务数量计算
                // 因为handleArrowModeCapture函数会为所有关节拍照
                if (jointOptions.length > 0) {
                    totalTasks += viewNames.length * jointOptions.length;
                } else {
                    totalTasks += viewNames.length; // 如果没有关节，则只计算视角数量
                }
            }
            else if (mode.needsJoints && jointOptions.length > 0) {
                totalTasks += viewNames.length * jointOptions.length;
            } else {
                totalTasks += viewNames.length;
            }
        }
        
        console.log(`Total ${totalTasks} capture tasks`);
        
        // 当前任务计数
        let currentTask = 0;
        
        // 处理每种渲染模式
        for (const renderMode of renderModes) {
            console.log(`Processing render mode: ${renderMode.mode}`);
            progressText.textContent = `Render mode: ${renderMode.mode}`;
            
            try {
                // 特殊处理arrow-sketch模式
                if (renderMode.mode === 'arrow-sketch') {
                    await handleArrowModeCapture(viewer, viewNames, renderMode, jointOptions, progressText, progressBar, modelInfo, currentTask, totalTasks);
                    continue; // 跳过后续处理
                }
                
                // 设置当前渲染模式按钮
                uncheckAllRenders();
                const renderButton = document.getElementById(renderMode.buttonId);
                if (renderButton) {
                    renderButton.classList.add('checked');
                    window.currentRenderMode = renderMode.mode;
                }
                
                // 设置背景颜色
                if (renderMode.blackBg) {
                    viewer.scene.background = new THREE.Color(0x000000);
                } else {
                    viewer.scene.background = null; // 透明背景
                }
                
                // 确保URDF模型可见
                if (viewer.robot) {
                    viewer.robot.visible = true;
                }
                
                // 更新渲染模式
                updateRenderMode(renderMode.mode, viewer, window.currentUrdfPath, window.currentRenderMode, window.defaultLight);
                
                // 等待渲染更新 - 减少等待时间
                await new Promise(resolve => setTimeout(resolve, 600)); // 从1000ms减少到600ms
                
                // 如果是需要处理关节的模式且有关节选项
                if (renderMode.needsJoints && jointOptions.length > 0) {
                    for (const joint of jointOptions) {
                        console.log(`Processing joint: ${joint}`);
                        
                        // 设置关节选择
                        if (jointSelect) {
                            jointSelect.value = joint;
                            // 触发change事件
                            const event = new Event('change');
                            jointSelect.dispatchEvent(event);
                            
                            // 等待UI更新
                            await new Promise(resolve => setTimeout(resolve, 500));
                        }
                        
                        // 捕获所有视角
                        for (let i = 0; i < viewNames.length; i++) {
                            const viewName = viewNames[i];
                            currentTask++;
                            
                            // 更新进度
                            const progress = (currentTask / totalTasks) * 100;
                            progressBar.style.width = `${progress}%`;
                            progressText.textContent = `${renderMode.mode} mode, joint: ${joint}, view: ${viewName} (${currentTask}/${totalTasks})`;
                            
                            // 捕获当前视角
                            await captureViewForGallery(viewer, viewName, joint, modelInfo.type, modelInfo.id, renderMode.mode);
                        }
                    }
                } else {
                    // 对于不需要处理关节的模式，直接捕获所有视角
                    for (let i = 0; i < viewNames.length; i++) {
                        const viewName = viewNames[i];
                        currentTask++;
                        
                        // 更新进度
                        const progress = (currentTask / totalTasks) * 100;
                        progressBar.style.width = `${progress}%`;
                        progressText.textContent = `${renderMode.mode} mode, view: ${viewName} (${currentTask}/${totalTasks})`;
                        
                        // 捕获当前视角
                        await captureViewForGallery(viewer, viewName, null, modelInfo.type, modelInfo.id, renderMode.mode);
                    }
                }
            } catch (modeError) {
                console.error(`Error processing render mode ${renderMode.mode}:`, modeError);
                progressText.textContent = `Render mode ${renderMode.mode} processing failed, skipping...`;
                await new Promise(resolve => setTimeout(resolve, 1000));
                // 继续下一个渲染模式
            }
        }

        // 恢复原始状态
        console.log('Capture complete, restoring original state');
        
        // 恢复视角
        viewer.camera.position.copy(originalView.position);
        viewer.controls.target.copy(originalView.target);
        viewer.controls.update();
        
        // 恢复背景
        viewer.scene.background = originalBackground;
        
        // 恢复渲染模式
        uncheckAllRenders();
        const originalRenderButton = document.getElementById(`${originalRenderMode}-render`);
        if (originalRenderButton) {
            originalRenderButton.classList.add('checked');
            window.currentRenderMode = originalRenderMode;
        }
        updateRenderMode(originalRenderMode, viewer, window.currentUrdfPath, originalRenderMode, window.defaultLight);
        
        // 更新进度指示器
        progressText.textContent = 'Capture Complete!';
        progressBar.style.width = '100%';
        
        // 3秒后移除进度指示器
        setTimeout(() => {
            document.body.removeChild(progressContainer);
        }, 3000);
        
    } catch (error) {
        console.error('Capture failed:', error);
        progressText.textContent = 'Capture failed: ' + error.message;
        progressBar.style.background = '#f44336';
        
        // 5秒后移除进度指示器
        setTimeout(() => {
            document.body.removeChild(progressContainer);
        }, 5000);
    }
    
    console.groupEnd();
}

// 处理arrow模式捕获的辅助函数
async function handleArrowModeCapture(viewer, viewNames, renderMode, jointOptions, progressText, progressBar, modelInfo, currentTask, totalTasks) {
    console.group('Processing arrow-sketch mode');
    progressText.textContent = 'Preparing arrow-sketch mode...';
    
            try {
                // 获取当前原始对象路径
                const originalObjPath = window.currentUrdfPath;
                if (!originalObjPath) {
                    console.error('Cannot get current URDF path');
            console.groupEnd();
                    return;
                }
                
        console.log(`Current URDF path: ${originalObjPath}`);
        console.log(`Model info:`, modelInfo);
        
        // 确保URDF模型不可见
                if (viewer.robot) {
            viewer.robot.visible = false;
                }
                
        // 设置当前渲染模式按钮
        uncheckAllRenders();
        const arrowSketchButton = document.getElementById('arrow-sketch');
        if (arrowSketchButton) {
            arrowSketchButton.classList.add('checked');
            window.currentRenderMode = 'arrow-sketch';
        }
        
        // 设置正确的背景 - 确保箭头模式使用透明背景
        viewer.scene.background = null;
        
        console.log(`Found ${jointOptions.length} joints for arrow-sketch mode:`, jointOptions);
        
        // 如果有关节，则为每个关节拍照（像segmentation模式一样）
        if (jointOptions && jointOptions.length > 0) {
            for (const jointId of jointOptions) {
                console.group(`Processing arrow-sketch mode for joint: ${jointId}`);
                progressText.textContent = `Preparing arrow-sketch for joint: ${jointId}`;
                
                try {
                    // 使用handleArrowJointIdChange真正切换到对应的关节
                    // 使用await等待箭头模型完全加载
                    console.log(`Changing to joint: ${jointId}`);
                    await handleArrowJointIdChange(viewer, originalObjPath, 'arrow-sketch', 'arrow-sketch', jointId);
                    
                    // 额外等待一些时间确保渲染稳定
                    console.log(`Waiting for rendering to stabilize...`);
                    await new Promise(resolve => setTimeout(resolve, 300));
        
                    // 为当前关节捕获所有视角
                    console.log(`Capturing all views for joint: ${jointId}`);
        for (let i = 0; i < viewNames.length; i++) {
            const viewName = viewNames[i];
            currentTask++;
            
            // 更新进度
            const progress = (currentTask / totalTasks) * 100;
            progressBar.style.width = `${progress}%`;
                        progressText.textContent = `arrow-sketch mode, joint: ${jointId}, view: ${viewName} (${currentTask}/${totalTasks})`;
            
                        // 捕获当前视角 - 传递jointId作为jointName参数
                        console.log(`Capturing view ${viewName} for joint ${jointId}`);
                        await captureViewForGallery(viewer, viewName, jointId, modelInfo.type, modelInfo.id, 'arrow-sketch');
        }
        
                    console.log(`Completed all views for joint: ${jointId}`);
                } catch (jointError) {
                    console.error(`Error processing joint ${jointId}:`, jointError);
                } finally {
                    // 在处理完一个关节后移除arrow模型，准备加载下一个
                    console.log(`Removing arrow model for joint: ${jointId}`);
                    removeArrowObject(viewer);
                    console.groupEnd(); // 关闭当前关节的日志组
                }
            }
        } else {
            // 如果没有关节选项，使用默认箭头（兼容旧行为）
            console.log('No joint options available, using default arrow');
            await new Promise((resolve) => {
                loadArrowModel(originalObjPath, viewer, () => {
                    // 应用arrow素描材质
                    applyArrowMaterial('arrow-sketch');
                    setTimeout(resolve, 500);
                });
            });
            
            // 对于没有关节的情况，直接捕获所有视角
            for (let i = 0; i < viewNames.length; i++) {
                const viewName = viewNames[i];
                currentTask++;
                
                // 更新进度
                const progress = (currentTask / totalTasks) * 100;
                progressBar.style.width = `${progress}%`;
                progressText.textContent = `arrow-sketch mode, view: ${viewName} (${currentTask}/${totalTasks})`;
                
                // 捕获当前视角 - 不传递jointName
                await captureViewForGallery(viewer, viewName, null, modelInfo.type, modelInfo.id, 'arrow-sketch');
        }
        
        // 移除arrow模型
        removeArrowObject(viewer);
        }
        
        console.log('Arrow-sketch mode processing complete');
        
    } catch (error) {
        console.error('Arrow-sketch mode processing failed:', error);
        progressText.textContent = 'Arrow-sketch mode processing failed, skipping...';
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // 确保URDF模型可见性恢复
        if (viewer.robot) {
            viewer.robot.traverse(obj => {
                if (obj.isMesh) {
                    obj.visible = true;
                }
            });
        }
        
        // 移除arrow模型
        removeArrowObject(viewer);
    }
    
    console.groupEnd(); // 关闭arrow-sketch处理的日志组
}

// 捕获所有模型函数 - 单独实现以避免修改原有captureGallery函数
async function captureAllModels(viewer) {
    console.group('Capturing All Models');
    
    // 获取当前模型，以便完成后恢复
    const originalModelInfo = getCurrentModelInfo();
    if (!originalModelInfo) {
        console.error('Cannot get current model information');
        console.groupEnd();
        return;
    }
    
    // 保存当前视角和渲染模式
    const originalView = {
        position: viewer.camera.position.clone(),
        target: viewer.controls.target.clone()
    };
    const originalRenderMode = getCurrentRenderMode();
    const originalBackground = viewer.scene.background;
    
    // 创建进度指示器
    const progressContainer = document.createElement('div');
    progressContainer.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: rgba(0, 0, 0, 0.8);
        color: white;
        padding: 20px;
        border-radius: 10px;
        z-index: 9999;
        text-align: center;
        min-width: 300px;
    `;
    
    const progressTitle = document.createElement('h3');
    progressTitle.textContent = 'All Models Multi-Mode Image Capture Progress';
    progressTitle.style.margin = '0 0 15px 0';
    
    const progressText = document.createElement('div');
    progressText.style.marginBottom = '10px';
    progressText.textContent = 'Preparing...';
    
    const progressBarContainer = document.createElement('div');
    progressBarContainer.style.cssText = `
        width: 100%;
        background-color: #333;
        border-radius: 5px;
        margin-bottom: 10px;
        overflow: hidden;
    `;
    
    const progressBar = document.createElement('div');
    progressBar.style.cssText = `
        height: 10px;
        background-color: #4CAF50;
        width: 0%;
        transition: width 0.3s;
    `;
    
    progressBarContainer.appendChild(progressBar);
    progressContainer.appendChild(progressTitle);
    progressContainer.appendChild(progressText);
    progressContainer.appendChild(progressBarContainer);
    
    document.body.appendChild(progressContainer);
    
    try {
        // 获取所有模型元素
        const modelElements = document.querySelectorAll('.model-item[urdf]');
        const models = Array.from(modelElements).map(element => ({
            id: element.textContent.trim(),
            urdfPath: element.getAttribute('urdf'),
            element: element
        }));
        
        progressText.textContent = `Found ${models.length} models to process`;
        console.log(`Found ${models.length} models to process`);
        
        // 用户确认（如果模型数量很多）
        if (models.length > 10) {
            const confirmResult = await new Promise(resolve => {
                const confirmBox = document.createElement('div');
                confirmBox.style.cssText = `
                    position: fixed;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    background: rgba(0, 0, 0, 0.9);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    z-index: 10000;
                    text-align: center;
                `;
                
                confirmBox.innerHTML = `
                    <h3>Confirm Batch Processing</h3>
                    <p>You are about to process ${models.length} 3D models, which may take a long time.</p>
                    <p>Are you sure to continue?</p>
                    <div style="display: flex; justify-content: center; gap: 10px; margin-top: 15px;">
                        <button id="confirm-yes" style="padding: 8px 20px; background: #4CAF50; border: none; border-radius: 4px; color: white; cursor: pointer;">Confirm</button>
                        <button id="confirm-no" style="padding: 8px 20px; background: #f44336; border: none; border-radius: 4px; color: white; cursor: pointer;">Cancel</button>
                    </div>
                `;
                
                document.body.appendChild(confirmBox);
                
                document.getElementById('confirm-yes').onclick = () => {
                    document.body.removeChild(confirmBox);
                    resolve(true);
                };
                
                document.getElementById('confirm-no').onclick = () => {
                    document.body.removeChild(confirmBox);
                    resolve(false);
                };
            });
            
            if (!confirmResult) {
                // 用户取消，清理并退出
                document.body.removeChild(progressContainer);
                console.log('User canceled batch processing');
                console.groupEnd();
                return;
            }
        }
        
        // 预先检查segmentation渲染模式按钮，以便在模型间切换时保持选中状态
        const segmentationButton = document.getElementById('segmentation-render');
        const segmentationActive = segmentationButton && segmentationButton.classList.contains('checked');
        const segmentationJointSelect = document.getElementById('segmentation-joint-select');
        const originalJointValue = segmentationJointSelect ? segmentationJointSelect.value : 'unselect';
        
        // 处理每个模型
        for (let i = 0; i < models.length; i++) {
            const model = models[i];
            
            try {
                // 更新进度
                const modelProgress = (i / models.length) * 100;
                progressBar.style.width = `${modelProgress}%`;
                
                progressText.textContent = `Processing model ${i + 1}/${models.length}: ${model.id}`;
                console.log(`Processing model ${i + 1}/${models.length}: ${model.id}`);
                
                // 点击模型元素以加载模型
                model.element.click();
                
                // 等待模型加载完成 - 增加等待时间以确保完全加载
                await new Promise(resolve => setTimeout(resolve, 3000));
                
                // 确保segmentation模式正确应用
                if (segmentationActive) {
                    console.log('Restoring segmentation render mode');
                    
                    // 先选中segmentation按钮
                    uncheckAllRenders();
                    segmentationButton.classList.add('checked');
                    
                    // 重新应用segmentation渲染模式
                    updateRenderMode('segmentation', viewer, window.currentUrdfPath, 'segmentation', window.defaultLight);
                    
                    // 如果之前选择了特定关节，尝试找到相似的关节并选择
                    if (segmentationJointSelect && originalJointValue !== 'unselect') {
                        // 重新填充关节选择下拉菜单
                        await new Promise(resolve => setTimeout(resolve, 500));
                        
                        // 尝试选择相同名称的关节，如果不存在则选择第一个关节
                        const jointOptions = Array.from(segmentationJointSelect.options);
                        const jointValues = jointOptions.map(option => option.value);
                        
                        if (jointValues.includes(originalJointValue)) {
                            segmentationJointSelect.value = originalJointValue;
                        } else if (jointValues.length > 1) {
                            // 跳过'unselect'，选择第一个实际关节
                            const firstJoint = jointValues.find(value => value !== 'unselect');
                            if (firstJoint) {
                                segmentationJointSelect.value = firstJoint;
                            }
                        }
                        
                        // 触发关节选择变更
                        const event = new Event('change');
                        segmentationJointSelect.dispatchEvent(event);
                        
                        // 给渲染模式一些时间来更新
                        await new Promise(resolve => setTimeout(resolve, 500));
                    }
                }
                
                // 执行gallery捕获
                await captureGallery(viewer);
                
                // 模型完成后暂停
                progressText.textContent = `Model ${i + 1}/${models.length}: ${model.id} processing complete`;
                await new Promise(resolve => setTimeout(resolve, 1000));
                
            } catch (modelError) {
                console.error(`Model processing failed: ${model.id}`, modelError);
                progressText.textContent = `Model ${i + 1}/${models.length}: ${model.id} processing failed`;
                await new Promise(resolve => setTimeout(resolve, 1000));
                // 继续下一个模型
            }
        }
        
        // 恢复到原始模型
        progressText.textContent = 'Restoring original model...';
        
        try {
            // 查找原始模型元素并点击
            const originalElement = document.querySelector(`.model-item[urdf="${originalModelInfo.urdfPath}"]`);
            if (originalElement) {
                originalElement.click();
                await new Promise(resolve => setTimeout(resolve, 2000));
            } else {
                console.error('Cannot find original model element:', originalModelInfo.urdfPath);
            }
        } catch (restoreError) {
            console.error('Failed to restore original model:', restoreError);
        }
        
        // 恢复视角和渲染模式
        viewer.camera.position.copy(originalView.position);
        viewer.controls.target.copy(originalView.target);
        viewer.controls.update();
        viewer.scene.background = originalBackground;
        
        // 恢复原始渲染模式
        uncheckAllRenders();
        const originalRenderButton = document.getElementById(`${originalRenderMode}-render`);
        if (originalRenderButton) {
            originalRenderButton.classList.add('checked');
            window.currentRenderMode = originalRenderMode;
        }
        updateRenderMode(originalRenderMode, viewer, window.currentUrdfPath, originalRenderMode, window.defaultLight);
        
        console.log('All models capture complete');
        progressText.textContent = 'All models capture complete!';
        progressBar.style.width = '100%';
        progressBar.style.background = '#4CAF50';
        
        // 3秒后移除进度指示器
        setTimeout(() => {
            document.body.removeChild(progressContainer);
        }, 3000);
        
    } catch (error) {
        console.error('Failed to capture all models:', error);
        progressText.textContent = 'Capture failed: ' + error.message;
        progressBar.style.background = '#f44336';
        
        // 5秒后移除进度指示器
        setTimeout(() => {
            document.body.removeChild(progressContainer);
        }, 5000);
    }
    
    console.groupEnd();
}

// 专门用于Gallery的视角拍摄函数，避免命名中的重复问题
async function captureViewForGallery(viewer, viewName, jointName, category, modelId, renderMode = 'segmentation') {
    console.group(`切换到视角: ${viewName}`);
    
    try {
    const viewPosition = window.viewPositions[viewName];
        
        // 检查视角数据是否有效
        if (!viewPosition || !viewPosition.position || !viewPosition.target) {
            console.error(`无效的视角数据: ${viewName}`);
            console.groupEnd();
            return;
        }
        
        // 保存当前相机位置用于验证
        const originalPosition = viewer.camera.position.clone();
        const originalTarget = viewer.controls.target.clone();
        
        console.log(`开始切换视角 ${viewName}`, {
            从: {
                position: originalPosition.toArray(),
                target: originalTarget.toArray()
            },
            到: {
                position: viewPosition.position.toArray(),
                target: viewPosition.target.toArray()
            }
        });
        
        // 模拟手动视图切换 - 直接使用viewerSetup.js中的applyView函数
        // 我们复制其实现而不是直接调用，以确保其工作方式完全一致
    viewer.camera.position.copy(viewPosition.position);
    viewer.controls.target.copy(viewPosition.target);
    viewer.controls.update();
    viewer.redraw();

        // 强制更新矩阵
        viewer.camera.updateProjectionMatrix();
        viewer.camera.updateMatrixWorld(true);
        
        // 等待视角更新 - 减少等待时间但确保足够更新
        console.log(`等待视角更新...`);
        await new Promise(resolve => setTimeout(resolve, 600)); // 从1000ms减少到600ms
        
        // 强制再次更新和渲染
        viewer.controls.update();
        viewer.redraw();
        await new Promise(resolve => setTimeout(resolve, 200)); // 从500ms减少到200ms
        
        // 验证视角是否真的发生变化
        const currentPosition = viewer.camera.position.clone();
        const distanceChanged = originalPosition.distanceTo(currentPosition);
        
        if (distanceChanged < 0.01) {
            console.warn(`视角可能未切换成功，尝试强制切换...`);
            
            // 更彻底的设置方式，确保成功切换
            viewer.camera.position.set(
                viewPosition.position.x,
                viewPosition.position.y,
                viewPosition.position.z
            );
            viewer.camera.lookAt(viewPosition.target);
            viewer.controls.target.set(
                viewPosition.target.x,
                viewPosition.target.y,
                viewPosition.target.z
            );
            viewer.controls.update();
            viewer.camera.updateProjectionMatrix();
            viewer.camera.updateMatrixWorld(true);
            viewer.redraw();
            
            // 再次等待和更新 - 减少时间但保证可靠性
            await new Promise(resolve => setTimeout(resolve, 600)); // 从1000ms减少到600ms
            viewer.redraw();
            
            // 最后检查
            const finalPosition = viewer.camera.position.clone();
            const finalDistanceChanged = originalPosition.distanceTo(finalPosition);
            console.log(`强制切换后位置变化: ${finalDistanceChanged}`);
            
            if (finalDistanceChanged < 0.01) {
                console.error(`视角切换失败: ${viewName}`);
                console.groupEnd();
                return;
            }
        }
        
        console.log(`视角切换成功: ${viewName}`, {
            位置变化: distanceChanged,
            最终相机位置: viewer.camera.position.toArray(),
            最终目标点: viewer.controls.target.toArray()
        });

        // 转换视角名称用于文件命名
    const viewLabel = viewName.replace(/-/g, '_');
    
    // 构建文件名后缀
    let jointSuffix = '';
    if (jointName) {
            // 对于任何渲染模式，只要有jointName就添加到文件名
        // 如果关节名已经包含"joint"前缀，则直接使用，否则添加前缀
        if (jointName.toLowerCase().startsWith('joint')) {
            jointSuffix = `_${jointName}`;
        } else {
            jointSuffix = `_joint_${jointName}`;
        }
    }
    
    // 构建完整文件名，加入渲染模式前缀
    const fileName = `${viewLabel}${jointSuffix}`;
    
        // 为日志添加模式和关节信息
        const logPrefix = jointName ? 
            `${renderMode}模式，关节${jointName}，视角${viewName}` :
            `${renderMode}模式，视角${viewName}`;
        
    try {
            // 确保在拍照前再次更新一次视图
            viewer.redraw();
            
        // 只下载图像，不下载JSON，不显示验证黑框
        await captureScreenshot(viewer, fileName, false, false, false, renderMode);
            console.log(`完成截图: ${logPrefix} => ${category}_${modelId}_${renderMode}_${fileName}`);
    } catch (err) {
            console.error(`截图失败: ${logPrefix} => ${category}_${modelId}_${renderMode}_${fileName}`, err);
    }
    
        // 在截图之间短暂暂停，避免过快请求 - 减少等待时间
        await new Promise(resolve => setTimeout(resolve, 150)); // 从300ms减少到150ms
        
    } catch (error) {
        console.error(`视角切换过程中发生错误:`, error);
    }
    
    console.groupEnd();
}

// 添加验证函数
async function validateCameraParams(viewer, screenshot, cameraParams) {
    console.group('Validating camera parameters');
    
    try {
        // 创建新的验证场景
        const validationScene = new THREE.Scene();
        
        // 创建相机并设置正确的坐标系
        const validationCamera = new THREE.PerspectiveCamera();
        validationCamera.up.set(0, 0, -1);  // 设置-Z为上方向，与原始场景一致

        // 从JSON重建相机参数
        validationCamera.fov = cameraParams.fov;
        validationCamera.aspect = cameraParams.aspect;
        validationCamera.near = cameraParams.near;
        validationCamera.far = cameraParams.far;
        validationCamera.position.set(
            cameraParams.position.x,
            cameraParams.position.y,
            cameraParams.position.z
        );
        validationCamera.quaternion.set(
            cameraParams.quaternion.x,
            cameraParams.quaternion.y,
            cameraParams.quaternion.z,
            cameraParams.quaternion.w
        );
        validationCamera.updateProjectionMatrix();

        // 添加坐标轴辅助（与原始场景使用相同的设置）
        const axesHelper = new THREE.AxesHelper(5);
        axesHelper.up = new THREE.Vector3(0, 0, -1);
        // 设置坐标轴颜色
        axesHelper.setColors(
            new THREE.Color(0xff0000), // X轴 - 红色
            new THREE.Color(0x00ff00), // Y轴 - 绿色
            new THREE.Color(0x0000ff)  // Z轴 - 蓝色
        );
        validationScene.add(axesHelper);

        // 重建场景 - 保持正确的模型方向和坐标系
        const validationRobot = viewer.robot.clone();
        // 确保模型使用正确的坐标系
        validationRobot.up = new THREE.Vector3(0, 0, -1);
        validationRobot.rotation.y = Math.PI;  // 旋转180度
        validationRobot.rotation.z = Math.PI*4; 
        validationRobot.rotation.x = Math.PI / 2; 
        validationRobot.updateMatrixWorld(true);
        validationScene.add(validationRobot);

        // 创建新的渲染器
        const validationRenderer = new THREE.WebGLRenderer({ 
            antialias: true,
            alpha: true
        });

        // 使用与原始截图相同的尺寸和比例
        validationRenderer.setSize(512, 512);
        validationRenderer.setPixelRatio(window.devicePixelRatio);

        // 计算保持宽高比的渲染尺寸
        const targetAspect = 1; // 512/512
        const currentAspect = cameraParams.aspect;
        let renderWidth, renderHeight;

        if (currentAspect > targetAspect) {
            // 当前画面更宽
            renderHeight = 512;
            renderWidth = Math.round(512 * currentAspect);
        } else {
            // 当前画面更高
            renderWidth = 512;
            renderHeight = Math.round(512 / currentAspect);
        }

        // 设置渲染尺寸
        validationRenderer.setSize(renderWidth, renderHeight, false);

        // 应用和原始场景相同的渲染设置
        if (cameraParams.renderSettings) {
            validationRenderer.outputEncoding = cameraParams.renderSettings.encoding;
            validationRenderer.gammaFactor = cameraParams.renderSettings.gammaFactor;
            validationRenderer.shadowMap.enabled = cameraParams.renderSettings.shadowMap;
        }

        // 重建灯光
        if (cameraParams.sceneInfo && cameraParams.sceneInfo.lightingSetup) {
            cameraParams.sceneInfo.lightingSetup.forEach(lightInfo => {
                const light = new THREE.DirectionalLight(lightInfo.color);
                light.position.set(
                    lightInfo.position[0],
                    lightInfo.position[1],
                    lightInfo.position[2]
                );
                light.intensity = lightInfo.intensity;
                validationScene.add(light);
            });
        }

        // 添加调试信息
        console.log('Validation Scene Setup:', {
            cameraUp: validationCamera.up.toArray(),
            robotUp: validationRobot.up.toArray(),
            sceneUp: validationScene.up ? validationScene.up.toArray() : 'undefined',
            cameraPosition: validationCamera.position.toArray(),
            cameraQuaternion: validationCamera.quaternion.toArray(),
            robotRotation: validationRobot.rotation.toArray()
        });

        // 设置背景色
        if (cameraParams.sceneInfo && cameraParams.sceneInfo.backgroundColor) {
            validationScene.background = new THREE.Color(cameraParams.sceneInfo.backgroundColor);
        }

        // 创建验证用的div
        const validationDiv = document.createElement('div');
        validationDiv.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: #000000;
            padding: 20px;
            border-radius: 10px;
            z-index: 1000;
            display: flex;
            flex-direction: column;
            gap: 10px;
            max-width: 800px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        `;

        // 添加图片对比
        const imagesDiv = document.createElement('div');
        imagesDiv.style.cssText = `
            display: flex;
            justify-content: space-between;
            gap: 20px;
        `;

        // 原始截图
        const originalImg = document.createElement('div');
        originalImg.innerHTML = `
            <p style="color: white; text-align: center;">Original Screenshot</p>
            <img src="${screenshot}" style="width: 350px; height: auto;">
        `;

        // 验证截图
        const validationImg = document.createElement('div');
        validationImg.innerHTML = `
            <p style="color: white; text-align: center;">Validation View</p>
            <img id="validation-screenshot" style="width: 350px; height: auto;">
        `;

        imagesDiv.appendChild(originalImg);
        imagesDiv.appendChild(validationImg);
        validationDiv.appendChild(imagesDiv);

        // 添加参数信息
        const paramsDiv = document.createElement('div');
        paramsDiv.style.color = 'white';
        paramsDiv.innerHTML = `
            <h3>Camera Parameters:</h3>
            <pre style="max-height: 200px; overflow-y: auto;">${JSON.stringify(cameraParams, null, 2)}</pre>
        `;
        validationDiv.appendChild(paramsDiv);

        // 添加按钮
        const buttonsDiv = document.createElement('div');
        buttonsDiv.style.cssText = `
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-top: 10px;
        `;

        const closeButton = document.createElement('button');
        closeButton.textContent = 'Close';
        closeButton.style.cssText = `
            padding: 8px 20px;
            cursor: pointer;
            background: #333333;
            color: white;
            border: none;
            border-radius: 4px;
            font-size: 14px;
        `;

        const retakeButton = document.createElement('button');
        retakeButton.textContent = 'Retake Screenshot';
        retakeButton.style.cssText = `
            padding: 8px 20px;
            cursor: pointer;
            background: #333333;
            color: white;
            border: none;
            border-radius: 4px;
            font-size: 14px;
        `;

        // 按钮悬停效果
        closeButton.onmouseover = () => closeButton.style.background = '#444444';
        closeButton.onmouseout = () => closeButton.style.background = '#333333';
        retakeButton.onmouseover = () => retakeButton.style.background = '#444444';
        retakeButton.onmouseout = () => retakeButton.style.background = '#333333';

        buttonsDiv.appendChild(closeButton);
        buttonsDiv.appendChild(retakeButton);
        validationDiv.appendChild(buttonsDiv);

        document.body.appendChild(validationDiv);

        // 渲染验证视图
        const renderValidationView = () => {
            validationRenderer.render(validationScene, validationCamera);
            
            // 创建一个512x512的canvas
            const canvas = document.createElement('canvas');
            canvas.width = 512;
            canvas.height = 512;
            const ctx = canvas.getContext('2d');
            
            // 计算居中位置
            const x = Math.round((512 - renderWidth) / 2);
            const y = Math.round((512 - renderHeight) / 2);
            
            // 绘制渲染结果到canvas上
            ctx.drawImage(validationRenderer.domElement, x, y, renderWidth, renderHeight);
            
            // 更新验证图像
            const validationImage = canvas.toDataURL('image/png');
            document.getElementById('validation-screenshot').src = validationImage;
        };

        // 初始渲染
        renderValidationView();

        // 按钮事件处理
        closeButton.onclick = () => {
            // 清理验证场景资源
            validationScene.traverse(object => {
                if (object.geometry) object.geometry.dispose();
                if (object.material) {
                    if (Array.isArray(object.material)) {
                        object.material.forEach(m => m.dispose());
                    } else {
                        object.material.dispose();
                    }
                }
            });
            validationRenderer.dispose();
            document.body.removeChild(validationDiv);
        };

        retakeButton.onclick = async () => {
            // 使用当前场景参数重新渲染验证视图
            renderValidationView();
        };

    } catch (error) {
        console.error('Error during validation:', error);
    }
    
    console.groupEnd();
}

// Update exports to remove getArrowPoints2DCoordinates
export { 
    addScreenshotControl, 
    addGalleryControl,
    captureScreenshot, 
    getArrowLines2DCoordinates,
    captureViewForGallery
};
