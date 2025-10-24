"use client";

import { useState, useEffect } from "react";
import Head from "next/head";
import { matchData } from "../lib/data";
import MatchCard from "../components/MatchCard";
import ScoreCard from "../components/ScoreCard";
import ThemeToggle from "../components/ThemeToggle";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { 
  Card, 
  CardContent, 
  CardHeader, 
  CardTitle 
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function Home() {
  const [showDetail, setShowDetail] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [activeTab, setActiveTab] = useState("live");

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return null;

  const fadeVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { 
        duration: 0.5,
        ease: "easeOut"
      }
    },
    exit: { 
      opacity: 0, 
      y: -20,
      transition: { 
        duration: 0.3,
        ease: "easeIn"
      }
    }
  };

  const staggerContainerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-b from-blue-50 to-white dark:from-gray-900 dark:to-gray-950">
      <Head>
        <title>IPL Match Analysis</title>
        <meta name="description" content="IPL T20 Cricket Match Statistics" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <header className="sticky top-0 z-10 backdrop-blur-xl bg-white/90 dark:bg-gray-900/90 border-b border-blue-100 dark:border-blue-900/30 shadow-sm">
        <div className="container mx-auto px-4 h-16 flex items-center justify-between">
          <motion.div 
            className="flex items-center space-x-3"
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3 }}
          >
            {showDetail && (
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                transition={{ type: "spring", stiffness: 500, damping: 30 }}
              >
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setShowDetail(false)}
                  className="mr-1 text-blue-600 dark:text-blue-400 hover:bg-blue-100 dark:hover:bg-blue-900/30"
                >
                  <ChevronLeft className="h-5 w-5" />
                </Button>
              </motion.div>
            )}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2, duration: 0.5 }}
              className="flex items-center"
            >
              <div className="h-8 w-8 mr-2 bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg shadow-md flex items-center justify-center">
                <svg 
                  className="h-5 w-5 text-white" 
                  viewBox="0 0 24 24" 
                  fill="none" 
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path 
                    d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" 
                    stroke="currentColor" 
                    strokeWidth="2"
                  />
                  <circle cx="12" cy="12" r="4" fill="currentColor" />
                </svg>
              </div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-blue-500 bg-clip-text text-transparent">
                IPL Match Analysis
              </h1>
            </motion.div>
          </motion.div>
          
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3, duration: 0.5 }}
          >
            <ThemeToggle />
          </motion.div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 flex-grow">
        <AnimatePresence mode="wait">
          {!showDetail ? (
            <motion.div
              key="home"
              initial="hidden"
              animate="visible"
              exit="exit"
              variants={fadeVariants}
              className="space-y-8"
            >
              <Tabs 
                defaultValue="live" 
                value={activeTab}
                onValueChange={setActiveTab}
                className="w-full"
              >
                <div className="flex items-center justify-between mb-6">
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2, duration: 0.4 }}
                  >
                    <TabsList className="bg-blue-100/50 dark:bg-blue-900/20">
                      <TabsTrigger 
                        value="live"
                        className="cursor-pointer dark:text-white text-gray-800 data-[state=active]:bg-white dark:data-[state=active]:bg-gray-800 data-[state=active]:text-blue-600 dark:data-[state=active]:text-blue-400"
                      >
                        Live Matches
                      </TabsTrigger>
                      <TabsTrigger 
                        value="upcoming"
                        className="cursor-pointer dark:text-white text-gray-800 data-[state=active]:bg-white dark:data-[state=active]:bg-gray-800 data-[state=active]:text-blue-600 dark:data-[state=active]:text-blue-400"
                      >
                        Upcoming
                      </TabsTrigger>
                      <TabsTrigger 
                        value="recent"
                        className="cursor-pointer dark:text-white text-gray-800 data-[state=active]:bg-white dark:data-[state=active]:bg-gray-800 data-[state=active]:text-blue-600 dark:data-[state=active]:text-blue-400"
                      >
                        Recent
                      </TabsTrigger>
                    </TabsList>
                  </motion.div>
                  
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.4, duration: 0.4 }}
                  >
                    <Badge variant="outline" className="bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 border-blue-200 dark:border-blue-800">
                      IPL 2025
                    </Badge>
                  </motion.div>
                </div>

                <TabsContent value="live" className="mt-0 space-y-6">
                  <motion.div 
                    variants={staggerContainerVariants}
                    initial="hidden"
                    animate="visible"
                    className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
                  >
                    <motion.div 
                      variants={fadeVariants}
                      whileHover={{ y: -5 }}
                      transition={{ type: "spring", stiffness: 300, damping: 15 }}
                    >
                      <MatchCard
                        match={matchData}
                        onClick={() => setShowDetail(true)}
                      />
                    </motion.div>
                    
                    {/* Additional match cards would be added here */}
                  </motion.div>
                </TabsContent>

                <TabsContent value="upcoming" className="mt-0">
                  <motion.div
                    variants={fadeVariants}
                    initial="hidden"
                    animate="visible"
                  >
                    <Card className="border-none shadow-lg overflow-hidden">
                      <CardHeader className="bg-gradient-to-r from-blue-600/10 to-blue-500/10 pb-2">
                        <CardTitle className="text-lg text-blue-700 dark:text-blue-300">
                          Upcoming Matches
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="p-6">
                        <motion.div 
                          className="text-center py-12"
                          initial={{ opacity: 0, scale: 0.9 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ duration: 0.5 }}
                        >
                          <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ delay: 0.3, duration: 0.5 }}
                          >
                            <svg
                              className="mx-auto h-16 w-16 text-blue-300 dark:text-blue-700"
                              fill="none"
                              viewBox="0 0 24 24"
                              stroke="currentColor"
                            >
                              <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={1}
                                d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
                              />
                            </svg>
                          </motion.div>
                          <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.5, duration: 0.5 }}
                          >
                            <h3 className="mt-4 text-lg font-medium text-gray-900 dark:text-gray-100">
                              No upcoming matches scheduled
                            </h3>
                            <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
                              Check back soon for the next exciting IPL fixtures
                            </p>
                            <Button 
                              variant="outline" 
                              className="mt-6 text-blue-600 dark:text-blue-400 border-blue-200 dark:border-blue-800 hover:bg-blue-50 dark:hover:bg-blue-900/20"
                            >
                              Subscribe to updates
                            </Button>
                          </motion.div>
                        </motion.div>
                      </CardContent>
                    </Card>
                  </motion.div>
                </TabsContent>

                <TabsContent value="recent" className="mt-0">
                  <motion.div
                    variants={fadeVariants}
                    initial="hidden"
                    animate="visible"
                  >
                    <Card className="border-none shadow-lg overflow-hidden">
                      <CardHeader className="bg-gradient-to-r from-blue-600/10 to-blue-500/10 pb-2">
                        <CardTitle className="text-lg text-blue-700 dark:text-blue-300">
                          Recently Concluded
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="p-6">
                        <div className="space-y-4">
                          <p className="text-center text-gray-500 dark:text-gray-400 py-8">
                            Recent match history will appear here.
                          </p>
                        </div>
                      </CardContent>
                    </Card>
                  </motion.div>
                </TabsContent>
              </Tabs>
              
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6, duration: 0.5 }}
                className="mt-12"
              >
                <Card className="border-none shadow-lg bg-gradient-to-r from-blue-600 to-blue-500 text-white overflow-hidden">
                  <CardContent className="p-6 flex flex-col md:flex-row items-center justify-between">
                    <div className="mb-4 md:mb-0">
                      <h3 className="text-xl font-bold mb-2">
                        Live IPL Match Analysis
                      </h3>
                      <p className="opacity-90">
                        Get insights and analysis for any IPL match in real time.
                      </p>
                    </div>
                    <Button 
                      variant="secondary" 
                      className="cursor-pointer bg-white hover:bg-gray-100 text-blue-600"
                    >
                      Analyze Match
                    </Button>
                  </CardContent>
                </Card>
              </motion.div>
            </motion.div>
          ) : (
            <motion.div
              key="detail"
              initial="hidden"
              animate="visible"
              exit="exit"
              variants={fadeVariants}
            >
              <ScoreCard match={matchData} />
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      <footer className="border-t border-blue-100 dark:border-blue-900/30 py-6 bg-white/80 dark:bg-gray-900/80 backdrop-blur-md">
        <div className="container mx-auto px-4">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2, duration: 0.5 }}
              className="flex items-center mb-4 md:mb-0"
            >
              <div className="h-6 w-6 mr-2 bg-gradient-to-br from-blue-500 to-blue-600 rounded-md shadow-sm flex items-center justify-center">
                <svg 
                  className="h-4 w-4 text-white" 
                  viewBox="0 0 24 24" 
                  fill="none" 
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path 
                    d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" 
                    stroke="currentColor" 
                    strokeWidth="2"
                  />
                  <circle cx="12" cy="12" r="4" fill="currentColor" />
                </svg>
              </div>
              <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
                © {new Date().getFullYear()} IPL Match Analysis
              </p>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3, duration: 0.5 }}
              className="flex space-x-6"
            >
              <a href="#" className="text-sm text-gray-500 hover:text-blue-600 dark:text-gray-400 dark:hover:text-blue-400 transition-colors">
                Terms
              </a>
              <a href="#" className="text-sm text-gray-500 hover:text-blue-600 dark:text-gray-400 dark:hover:text-blue-400 transition-colors">
                Privacy
              </a>
              <a href="#" className="text-sm text-gray-500 hover:text-blue-600 dark:text-gray-400 dark:hover:text-blue-400 transition-colors">
                Support
              </a>
            </motion.div>
          </div>
        </div>
      </footer>
    </div>
  );
}