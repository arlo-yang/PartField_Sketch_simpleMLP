/* globals */
import URDFManipulator from '../../src/urdf-manipulator-element.js';
import { initialModels } from './models-config.js';
import { setColor } from './utils/visualManager.js';
import { initializeCategories, initURDFEvents } from './utils/modelManager.js';
import { initViewerSetup } from './utils/viewerSetup.js';

import { initUrdfInfo } from './utils/urdfInfo.js';


// 注册自定义元素
customElements.define('urdf-viewer', URDFManipulator);

// 确保setColor函数可用
const safeSetColor = (color) => {
    try {
        if (typeof setColor === 'function') {
            setColor(color);
        } else {
            document.body.style.backgroundColor = color || '#263238';
        }
    } catch (error) {
        console.error('Error setting color:', error);
        document.body.style.backgroundColor = '#263238';
    }
};

// 初始化函数
async function init() {
    try {
        // 添加错误捕获，抑制特定的Three.js错误
        setupErrorSuppression();
        
        // 确保背景色正确设置
        document.body.style.backgroundColor = '#263238';
        
        const viewer = document.querySelector('urdf-viewer');
        if (!viewer) {
            console.error('Failed to find urdf-viewer element');
            return;
        }

        // 确保控制面板初始状态
        const controlsPanel = document.getElementById('controls');
        if (controlsPanel) {
            controlsPanel.classList.add('hidden');
        }

        // 初始化场景设置
        initViewerSetup(viewer);

        // 初始化 URDF 事件监听
        initURDFEvents(viewer);
        
        // 初始化URDF信息显示功能 - 只在需要时加载和执行
        const currentMode = window.currentRenderMode || 'default';
        if (currentMode === 'segmentation') {
            initUrdfInfo(viewer);
        } else {
            // 监听渲染模式变化，以便在切换到segmentation模式时初始化
            window.addEventListener('renderModeChanged', (e) => {
                const mode = e.detail?.mode || 'default';
                if (mode === 'segmentation' && !window.urdfInfoInitialized) {
                    window.urdfInfoInitialized = true;
                    initUrdfInfo(viewer);
                }
            });
        }
        

        // 更新模型分类列表
        await initializeCategories(viewer);

        // 设置初始模型
        if (initialModels && initialModels.length > 0) {
            const firstModel = initialModels[0];
            window.currentUrdfPath = firstModel.urdf;
            console.log('设置初始URDF路径:', window.currentUrdfPath);
            
            // 确保正确设置包路径
            if (!/^https?:\/\//.test(firstModel.urdf)) {
                viewer.package = firstModel.urdf.split('/').slice(0, -2).join('/');
            }
            
            viewer.up = '-Z';
            viewer.urdf = firstModel.urdf;
            safeSetColor(firstModel.color);

            // 添加加载状态监听
            viewer.addEventListener('urdf-processed', () => {
                console.log('Initial URDF processed');
                viewer.redraw();
            }, { once: true });
        }

        // 添加错误处理
        viewer.addEventListener('error', (e) => {
            console.error('URDF Viewer Error:', e.detail);
        });

        // 添加加载完成处理
        viewer.addEventListener('load', () => {
            console.log('URDF loaded successfully');
            viewer.redraw();
        });

    } catch (error) {
        console.error('Initialization error:', error);
        // 确保即使出错也设置背景色
        document.body.style.backgroundColor = '#263238';
    }
}

// 设置错误抑制函数
function setupErrorSuppression() {
    // 保存原始的console.error方法
    const originalConsoleError = console.error;
    
    // 需要抑制的错误消息模式
    const suppressPatterns = [
        'THREE.Sprite: "Raycaster.camera" needs to be set',
        'Cannot read properties of null (reading \'matrixWorld\')'
    ];
    
    // 记录最近的错误，避免重复打印
    const recentErrors = new Set();
    const ERROR_TIMEOUT = 5000; // 5秒内相同错误只显示一次
    
    // 替换console.error方法
    console.error = function(...args) {
        // 检查是否为需要抑制的错误
        const errorString = args.join(' ');
        const shouldSuppress = suppressPatterns.some(pattern => 
            errorString.includes(pattern)
        );
        
        if (shouldSuppress) {
            // 检查是否最近已显示过此错误
            if (!recentErrors.has(errorString)) {
                // 第一次出现时仍然显示，但添加说明
                originalConsoleError.apply(console, [...args, '(此错误将被抑制以减少控制台噪音)']);
                
                // 添加到最近错误集合
                recentErrors.add(errorString);
                
                // 设置超时后从集合中移除
                setTimeout(() => {
                    recentErrors.delete(errorString);
                }, ERROR_TIMEOUT);
            }
            // 否则完全抑制
            return;
        }
        
        // 对于其他错误，使用原始方法
        originalConsoleError.apply(console, args);
    };
    
    // 捕获并抑制特定的未捕获异常
    window.addEventListener('error', function(event) {
        const errorMsg = event.message || '';
        const errorStack = event.error?.stack || '';
        
        // 检查是否为需要抑制的错误
        const shouldSuppress = suppressPatterns.some(pattern => 
            errorMsg.includes(pattern) || errorStack.includes(pattern)
        );
        
        if (shouldSuppress) {
            // 阻止错误传播到控制台
            event.preventDefault();
            return false;
        }
    }, true);
}

// 确保 DOM 加载完成后再初始化
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

// 添加一个全局的错误处理，确保背景色始终正确
window.addEventListener('error', () => {
    document.body.style.backgroundColor = '#263238';
});
