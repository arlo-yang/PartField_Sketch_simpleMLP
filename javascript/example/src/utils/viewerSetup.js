import * as THREE from 'three';
import { registerDragEvents } from '../dragAndDrop.js';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { ColladaLoader } from 'three/examples/jsm/loaders/ColladaLoader.js';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';
import { addScreenshotControl, addGalleryControl } from './seg_screenshot.js';
import { applyProjectionMaterial, updateRenderMode, setupExportFaceIdButton } from './visualManager.js';
import { initUIControls } from './uiControls.js';

// 定义不同的视角参数
export const viewPositions = (() => {
    // 辅助函数：根据球面坐标创建相机位置
    const createCameraPosition = (distance, phi, theta) => {
        return {
            position: new THREE.Vector3(
                distance * Math.cos(phi) * Math.cos(theta),
                distance * Math.sin(phi),
                distance * Math.cos(phi) * Math.sin(theta)
            ),
            target: new THREE.Vector3(0, 0, 0)
        };
    };

    // 统一的相机距离
    const distance = 2.50;
    
    // 定义常用角度
    const angles = {
        phi: {
            top: Math.PI / 2,     // 90度仰角
            highAngle: Math.PI / 3,  // 60度仰角
            mediumUp: Math.PI / 4,   // 45度仰角
            level: 0,               // 水平视角
            mediumDown: -Math.PI / 4, // -45度俯角
            lowAngle: -Math.PI / 3,  // -60度俯角
            bottom: -Math.PI / 2    // -90度俯角
        },
        theta: {
            left90: Math.PI / 2,    // 90度左偏
            left45: Math.PI / 4,    // 45度左偏
            center: 0,              // 正面
            right45: -Math.PI / 4,  // -45度右偏
            right90: -Math.PI / 2   // -90度右偏
        }
    };

    return {
        // 保留的视角
        'top-left': createCameraPosition(distance, angles.phi.mediumUp, angles.theta.left45),
        'left': createCameraPosition(distance, angles.phi.level, angles.theta.left45),
        'bottom-left': createCameraPosition(distance, angles.phi.mediumDown, angles.theta.left45),
        
        // 中间视角（从上到下）
        'top-center': createCameraPosition(distance, angles.phi.mediumUp, angles.theta.center),
        'center': createCameraPosition(distance, angles.phi.level, angles.theta.center),
        'bottom-center': createCameraPosition(distance, angles.phi.mediumDown, angles.theta.center),
        
        // 右侧视角（从上到下）
        'top-right': createCameraPosition(distance, angles.phi.mediumUp, angles.theta.right45),
        'right': createCameraPosition(distance, angles.phi.level, angles.theta.right45),
        'bottom-right': createCameraPosition(distance, angles.phi.mediumDown, angles.theta.right45)
    };
})();

// 应用视角的函数
export function applyView(viewer, view) {
    viewer.camera.position.copy(view.position);
    viewer.controls.target.copy(view.target);
    viewer.controls.update();
    viewer.redraw();
}

// 将预设视角保存到全局，方便其它模块使用
window.viewPositions = viewPositions;

export function initViewerSetup(viewer) {
  console.group('Initializing Viewer Setup');
  
  if (!viewer) {
    console.error('Viewer not found in initViewerSetup');
    console.groupEnd();
    return;
  }

  console.log('Current viewer state:', {
    scene: !!viewer.scene,
    camera: !!viewer.camera,
    controls: !!viewer.controls,
    currentUrdfPath: window.currentUrdfPath
  });

  console.log('Initializing viewer setup with currentUrdfPath:', window.currentUrdfPath);

  // 添加Raycaster修复
  fixRaycasterCamera(viewer);

  // 添加默认定向光
  const defaultLight = new THREE.DirectionalLight(0xffffff, 1.2);
  defaultLight.position.set(2, 2, 2);
  viewer.scene.add(defaultLight);
  window.defaultLight = defaultLight; // 保存全局供 uiControls 使用

  // 添加环境光
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
  viewer.scene.add(ambientLight);

  // 初始化 UI 控制器（在添加灯光之后）
  console.log('Initializing UI controls...');
  initUIControls(viewer, defaultLight);

  // 配置 mesh loader
  viewer.loadMeshFunc = (path, manager, done) => {
    const ext = path.split(/\./g).pop().toLowerCase();
    
    const onError = (err) => {
      console.error('Error loading mesh:', err);
      done(null, err);
    };

    const onProgress = (xhr) => {
      console.log(`${(xhr.loaded / xhr.total * 100)}% loaded`);
    };

    try {
      switch (ext) {
        case 'ply':
          new PLYLoader(manager).load(
            path,
            (geometry) => {
              const material = new THREE.MeshPhongMaterial();
              const mesh = new THREE.Mesh(geometry, material);
              done(mesh);
            },
            onProgress,
            onError
          );
          break;
        case 'gltf':
        case 'glb':
          new GLTFLoader(manager).load(
            path,
            (gltf) => done(gltf.scene),
            onProgress,
            onError
          );
          break;
        case 'obj':
          new OBJLoader(manager).load(
            path,
            (obj) => done(obj),
            onProgress,
            onError
          );
          break;
        case 'dae':
          new ColladaLoader(manager).load(
            path,
            (collada) => done(collada.scene),
            onProgress,
            onError
          );
          break;
        case 'stl':
          new STLLoader(manager).load(
            path,
            (geometry) => {
              const material = new THREE.MeshPhongMaterial();
              const mesh = new THREE.Mesh(geometry, material);
              done(mesh);
            },
            onProgress,
            onError
          );
          break;
        default:
          onError(new Error(`Unsupported file extension: ${ext}`));
      }
    } catch (error) {
      onError(error);
    }
  };

  // 设置 package 路径
  if (/javascript\/example\/bundle/i.test(window.location)) {
    viewer.package = '../../../urdf';
  } else {
    // 确保设置正确的包路径
    viewer.package = '/urdf';
  }

  // 注册拖拽事件
  registerDragEvents(viewer, () => {
    if (window.setColor) {
      window.setColor('#263238');
    } else {
      document.body.style.backgroundColor = '#263238';
    }
    if (window.updateList) {
      window.updateList(viewer);
    }
  });

  // URDF 处理完成后，根据当前选中的渲染按钮更新渲染模式
  viewer.addEventListener('urdf-processed', () => {
    const normalRender = document.getElementById('normal-render');
    const depthRender = document.getElementById('depth-render');
    const wireframeRender = document.getElementById('wireframe-render');
    const arrowDefault = document.getElementById('arrow-default');
    const arrowWireframe = document.getElementById('arrow-wireframe');
    const arrowSketch = document.getElementById('arrow-sketch');
    const defaultObjRender = document.getElementById('default-obj-render');
    let mode = 'default';
    if (normalRender && normalRender.classList.contains('checked')) mode = 'normal';
    if (depthRender && depthRender.classList.contains('checked')) mode = 'depth';
    if (wireframeRender && wireframeRender.classList.contains('checked')) mode = 'wireframe';
    if (arrowDefault && arrowDefault.classList.contains('checked')) mode = 'arrow-default';
    if (arrowWireframe && arrowWireframe.classList.contains('checked')) mode = 'arrow-wireframe';
    if (arrowSketch && arrowSketch.classList.contains('checked')) mode = 'arrow-sketch';
    if (defaultObjRender && defaultObjRender.classList.contains('checked')) mode = 'default-obj';
    updateRenderMode(mode, viewer, window.currentUrdfPath, window.currentRenderMode, defaultLight);
    
    // 触发渲染模式变化事件，确保URDF面板可见性正确
    const event = new CustomEvent('renderModeChanged', {
      detail: { mode: window.currentRenderMode }
    });
    window.dispatchEvent(event);
  });

  // 当 URDF 处理完后更新关节角度（2D UI 部分）
  viewer.addEventListener('urdf-processed', () => updateAngles(viewer));
  updateLoop();

  // 初始化控制器
  console.log('Initializing controllers...');
  const axesController = addAxesControl(viewer);
  const viewController = addViewControl(viewer);

  // 保存到全局以供其他模块使用
  window.axesController = axesController;
  window.viewController = viewController;

  console.log('Controllers initialized:', {
    axesController: !!axesController,
    viewController: !!viewController
  });

  // 添加错误处理
  if (!axesController) {
    console.error('Failed to initialize axes controller');
  }
  if (!viewController) {
    console.error('Failed to initialize view controller');
  }

  // 初始化截图控制器
  addScreenshotControl(viewer, window.currentUrdfPath);
  addGalleryControl(viewer, window.currentUrdfPath);
  
  // 设置相机参数
  viewer.camera.near = 0.1;
  viewer.camera.far = 15;
  viewer.camera.updateProjectionMatrix();

  // 添加错误处理
  viewer.addEventListener('error', (e) => {
    console.error('Viewer error:', e.detail);
  });

  // 添加 toggle controls 功能
  const toggleControlsButton = document.getElementById('toggle-controls');
  const controlsPanel = document.getElementById('controls');
  
  if (toggleControlsButton && controlsPanel) {
    console.log('Found controls elements');
    
    toggleControlsButton.addEventListener('click', () => {
        console.group('Toggle Controls Clicked');
        try {
            const isHidden = controlsPanel.classList.contains('hidden');
            console.log('Controls current state:', { isHidden });
            
            controlsPanel.classList.toggle('hidden');
            console.log('Controls toggled successfully');
        } catch (error) {
            console.error('Error toggling controls:', error);
        }
        console.groupEnd();
    });
    
    // 确保初始状态是隐藏的
    controlsPanel.classList.add('hidden');
  } else {
    console.error('Controls elements not found:', {
        toggleButton: !!toggleControlsButton,
        panel: !!controlsPanel
    });
  }

  // 在相机控制器初始化后添加事件监听
  if (viewer.controls) {
    // 添加相机控制器变化事件监听
    viewer.controls.addEventListener('change', () => {
        // 触发自定义事件，通知相机已变化
        const event = new CustomEvent('camera-change');
        viewer.dispatchEvent(event);
    });
  }

  console.log('Viewer setup completed');
  console.groupEnd();
}

// 辅助函数

function updateAngles(viewer) {
  if (!viewer.setJointValue) return;
  const resetJointValues = viewer.angles;
  for (const name in resetJointValues) {
    resetJointValues[name] = 0;
  }
  viewer.setJointValues(resetJointValues);
}

function updateLoop() {
  requestAnimationFrame(updateLoop);
}

function addAxesControl(viewer) {
    console.group('Initializing Axes Control');
    const axesButton = document.getElementById('axes-toggle');
    
    if (!axesButton) {
        console.error('Axes control button not found in DOM');
        console.groupEnd();
        return null;
    }

    // 创建坐标轴辅助对象
    const axesHelper = new THREE.AxesHelper(5);
    axesHelper.setColors(
        new THREE.Color(0xff0000), // X轴 - 红色
        new THREE.Color(0x00ff00), // Y轴 - 绿色
        new THREE.Color(0x0000ff)  // Z轴 - 蓝色
    );

    // 创建标签精灵
    function createAxisLabel(text, position, color) {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = 64;
        canvas.height = 32;

        context.font = 'Bold 24px Arial';
        context.fillStyle = color;
        context.textAlign = 'center';
        context.textBaseline = 'middle';
        context.fillText(text, 32, 16);

        const texture = new THREE.CanvasTexture(canvas);
        const spriteMaterial = new THREE.SpriteMaterial({ 
            map: texture,
            sizeAttenuation: false
        });

        const sprite = new THREE.Sprite(spriteMaterial);
        sprite.position.copy(position);
        sprite.scale.set(0.5, 0.25, 1);
        return sprite;
    }

    // 添加标签到坐标轴
    const labels = [
        { text: '+X', position: new THREE.Vector3(5.5, 0, 0), color: '#ff0000' },
        { text: '+Y', position: new THREE.Vector3(0, 5.5, 0), color: '#00ff00' },
        { text: '+Z', position: new THREE.Vector3(0, 0, 5.5), color: '#0000ff' }
    ];

    // 创建标签组
    const labelGroup = new THREE.Group();
    labels.forEach(label => {
        const sprite = createAxisLabel(label.text, label.position, label.color);
        labelGroup.add(sprite);
    });

    // 将标签组添加到坐标轴
    axesHelper.add(labelGroup);

    // 确保初始状态下坐标轴是隐藏的
    axesHelper.visible = false;

    // 将坐标轴添加到场景中
    viewer.scene.add(axesHelper);

    // 点击事件处理
    axesButton.addEventListener('click', () => {
        const isActive = axesButton.classList.contains('checked');
        if (!isActive) {
            axesButton.classList.add('checked');
            axesHelper.visible = true;
        } else {
            axesButton.classList.remove('checked');
            axesHelper.visible = false;
        }
        viewer.redraw();
    });

    return {
        show: () => {
            axesButton.classList.add('checked');
            axesHelper.visible = true;
            viewer.redraw();
        },
        hide: () => {
            axesButton.classList.remove('checked');
            axesHelper.visible = false;
            viewer.redraw();
        },
        toggle: () => axesButton.click(),
        isVisible: () => axesButton.classList.contains('checked')
    };
}

function addViewControl(viewer) {
  console.group('Initializing View Control');
  const viewButton = document.getElementById('view-toggle');
  const viewSelect = document.getElementById('view-select');
  
  if (!viewButton || !viewSelect) {
    console.error('View control elements not found:', {
      viewButton: !!viewButton,
      viewSelect: !!viewSelect
    });
    console.groupEnd();
    return null;
  }
  console.log('Found view controls:', { viewButton, viewSelect });

  let savedView = null;

  // 保存当前视角的函数
  function saveCurrentView() {
    try {
      const view = {
        position: viewer.camera.position.clone(),
        target: viewer.controls.target.clone()
      };
      console.log('Saved view:', view);
      return view;
    } catch (error) {
      console.error('Error saving view:', error);
      return null;
    }
  }

  // 应用视角的函数
  function applyViewPosition(view) {
    console.group('Applying View Position');
    if (!view) {
      console.warn('No view provided to apply');
      console.groupEnd();
      return;
    }
    
    try {
      console.log('Current camera state:', {
        position: viewer.camera.position.toArray(),
        controls: !!viewer.controls,
        target: viewer.controls ? viewer.controls.target.toArray() : null
      });

      viewer.camera.position.copy(view.position);
      if (viewer.controls) {
        viewer.controls.target.copy(view.target);
        viewer.controls.update();
        console.log('Updated camera and controls');
      } else {
        console.warn('No controls available');
      }
      viewer.redraw();
    } catch (error) {
      console.error('Error applying view position:', error);
    }
    console.groupEnd();
  }

  // 点击事件处理
  viewButton.addEventListener('click', (e) => {
    console.group('View Button Clicked');
    const isActive = viewButton.classList.contains('checked');
    console.log('Current state:', { isActive, buttonClasses: viewButton.classList.toString() });
    
    try {
      if (!isActive) {
        viewButton.classList.add('checked');
        savedView = saveCurrentView();
        
        const selectedView = window.viewPositions[viewSelect.value];
        console.log('Selected view:', { viewName: viewSelect.value, view: selectedView });
        if (selectedView) {
          applyViewPosition(selectedView);
        }
      } else {
        viewButton.classList.remove('checked');
        if (savedView) {
          applyViewPosition(savedView);
        }
      }
    } catch (error) {
      console.error('Error in view button click handler:', error);
    }
    console.groupEnd();
  });

  // 视角选择变化事件
  viewSelect.addEventListener('change', (e) => {
    console.group('View Selection Changed');
    try {
      if (viewButton.classList.contains('checked')) {
        const selectedView = window.viewPositions[viewSelect.value];
        console.log('New view selected:', { viewName: viewSelect.value, view: selectedView });
        if (selectedView) {
          applyViewPosition(selectedView);
        }
      } else {
        console.log('View not active, ignoring selection change');
      }
    } catch (error) {
      console.error('Error in view selection handler:', error);
    }
    console.groupEnd();
  });

  console.log('View control initialized successfully');
  console.groupEnd();
  return {
    show: () => {
      console.group('View Show Called');
      try {
        viewButton.classList.add('checked');
        savedView = saveCurrentView();
        const selectedView = window.viewPositions[viewSelect.value];
        if (selectedView) {
          applyViewPosition(selectedView);
        }
        console.log('View shown successfully');
      } catch (error) {
        console.error('Error showing view:', error);
      }
      console.groupEnd();
    },
    hide: () => {
      console.group('View Hide Called');
      try {
        viewButton.classList.remove('checked');
        if (savedView) {
          applyViewPosition(savedView);
        }
        console.log('View hidden successfully');
      } catch (error) {
        console.error('Error hiding view:', error);
      }
      console.groupEnd();
    },
    toggle: () => {
      console.log('View Toggle Called');
      viewButton.click();
    },
    isVisible: () => {
      const visible = viewButton.classList.contains('checked');
      console.log('View visibility checked:', visible);
      return visible;
    }
  };
}

// 修复Raycaster相机问题
function fixRaycasterCamera(viewer) {
    try {
        // 确保viewer有场景和相机
        if (!viewer.scene || !viewer.camera) {
            console.warn('Cannot fix Raycaster: scene or camera not initialized');
            
            // 添加场景加载后的回调，以便稍后应用修复
            viewer.addEventListener('load', () => {
                console.log('Scene loaded, applying delayed raycaster fixes');
                setTimeout(() => fixRaycasterCamera(viewer), 500);
            }, { once: true });
            
            return;
        }
        
        // 修复全局THREE.Raycaster
        const originalIntersectObject = THREE.Raycaster.prototype.intersectObject;
        THREE.Raycaster.prototype.intersectObject = function(object, recursive, optionalTarget) {
            // 确保相机已设置
            if (!this.camera && viewer.camera) {
                this.camera = viewer.camera;
            }
            
            // 如果对象是Sprite且我们没有相机，跳过而不是出错
            if (object.isSprite && !this.camera) {
                return optionalTarget || [];
            }
            
            try {
                return originalIntersectObject.call(this, object, recursive, optionalTarget);
            } catch (error) {
                // 忽略matrixWorld错误
                if (error.message && error.message.includes('matrixWorld')) {
                    return optionalTarget || [];
                }
                throw error; // 重新抛出其他错误
            }
        };
        
        // 处理现有Sprite对象
        viewer.scene.traverse((object) => {
            if (object.isSprite) {
                // 确保Sprite对象的raycast方法不会出错
                const originalRaycast = object.raycast;
                object.raycast = function(raycaster, intersects) {
                    // 确保raycaster有相机引用
                    if (!raycaster.camera) {
                        raycaster.camera = viewer.camera;
                    }
                    
                    // 如果仍然没有相机，则安全地返回
                    if (!raycaster.camera) return;
                    
                    // 调用原始方法
                    try {
                        originalRaycast.call(this, raycaster, intersects);
                    } catch (e) {
                        // 静默错误
                    }
                };
            }
        });
        
        // 添加场景变化监听，以修复新添加的Sprite
        viewer.scene.addEventListener('childadded', (event) => {
            const object = event.child;
            if (object.isSprite) {
                // 同样修复新添加的Sprite
                const originalRaycast = object.raycast;
                object.raycast = function(raycaster, intersects) {
                    if (!raycaster.camera) {
                        raycaster.camera = viewer.camera;
                    }
                    if (!raycaster.camera) return;
                    
                    try {
                        originalRaycast.call(this, raycaster, intersects);
                    } catch (e) {
                        // 静默错误
                    }
                };
            }
        });
        
        // 修复控制器
        if (viewer.controls) {
            // 为控制器添加错误处理
            const originalHandleMouseMove = viewer.controls._mousemove;
            if (typeof originalHandleMouseMove === 'function') {
                viewer.controls._mousemove = function(e) {
                    try {
                        originalHandleMouseMove.call(this, e);
                    } catch (error) {
                        // 忽略特定错误
                        if (!error.message || !error.message.includes('matrixWorld')) {
                            console.error('MouseMove error:', error);
                        }
                    }
                };
            }
            
            // 如果有dragControls也修复它
            if (viewer.controls._dragControls) {
                const dragControls = viewer.controls._dragControls;
                
                // 修复update方法
                if (typeof dragControls.update === 'function') {
                    const originalUpdate = dragControls.update;
                    dragControls.update = function(event) {
                        // 确保raycaster有相机
                        if (this.raycaster && !this.raycaster.camera) {
                            this.raycaster.camera = viewer.camera;
                        }
                        
                        // 安全调用原始方法
                        try {
                            originalUpdate.call(this, event);
                        } catch (e) {
                            // 忽略特定错误
                        }
                    };
                }
                
                // 修复moveRay方法
                if (typeof dragControls.moveRay === 'function') {
                    const originalMoveRay = dragControls.moveRay;
                    dragControls.moveRay = function() {
                        // 确保raycaster有相机
                        if (this.raycaster && !this.raycaster.camera) {
                            this.raycaster.camera = viewer.camera;
                        }
                        
                        // 安全调用原始方法
                        try {
                            originalMoveRay.call(this);
                        } catch (e) {
                            // 忽略特定错误
                        }
                    };
                }
            }
        }
        
        console.log('Applied Raycaster camera fixes successfully');
    } catch (error) {
        console.warn('Error applying Raycaster fixes:', error);
    }
}
