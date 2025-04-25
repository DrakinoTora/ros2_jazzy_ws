from setuptools import find_packages, setup

package_name = 'ball_det_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/ball_det_package']),
        ('share/ball_det_package', [
            'package.xml',
            'ball_det_package/sample3.mp4'
        ]),
    ],
    install_requires=['setuptools', 'opencv-python', 'numpy'],
    zip_safe=True,
    maintainer='azma',
    maintainer_email='azma@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ball_det_node = ball_det_package.ball_det_node:main'
        ],
    },
)
