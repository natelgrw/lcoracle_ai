import React from 'react';
import { motion } from 'framer-motion';

const InteractiveBackground: React.FC = () => {
    return (
        <div className="absolute inset-0 overflow-hidden pointer-events-none z-0">
            {/* Blob 1: Pink-ish */}
            <motion.div className="absolute top-1/3 right-1/4">
                <motion.div
                    animate={{
                        x: [0, -180, 120, 0],
                        y: [0, 140, -100, 0],
                        scale: [1, 1.3, 0.8, 1],
                        opacity: [0.4, 0.5, 0.3],
                    }}
                    transition={{
                        duration: 10,
                        repeat: Infinity,
                        ease: "easeInOut",
                        delay: 1
                    }}
                    className="w-[32rem] h-[32rem] bg-pink-300/40 rounded-full mix-blend-multiply filter blur-[80px] opacity-50"
                />
            </motion.div>

            {/* Blob 2: Indigo */}
            <motion.div className="absolute bottom-1/4 left-1/3">
                <motion.div
                    animate={{
                        x: [0, 160, -160, 0],
                        y: [0, 100, -120, 0],
                        scale: [1, 0.9, 1.2, 1],
                        opacity: [0.3, 0.5, 0.3],
                    }}
                    transition={{
                        duration: 10,
                        repeat: Infinity,
                        ease: "easeInOut",
                        delay: 1
                    }}
                    className="w-96 h-96 bg-indigo-300/40 rounded-full mix-blend-multiply filter blur-[64px] opacity-50"
                />
            </motion.div>
        </div>
    );
};

export default InteractiveBackground;
