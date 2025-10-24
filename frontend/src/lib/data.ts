import { Match, Player } from "../types/match";

// Match data for IPL 2021 Match #55
export const matchData: Match = {
  id: "ipl-2021-55",
  date: "2021-10-08",
  venue: "Zayed Cricket Stadium, Abu Dhabi",
  matchType: "T20",
  matchNumber: "55 of 60",
  result: "Mumbai Indians won by 42 runs",
  winningTeam: "Mumbai Indians",
  winMargin: "42 runs",
  innings: [
    {
      team: {
        id: "mi",
        name: "Mumbai Indians",
        shortName: "MI",
        logo: "/logos/mi.png",
        players: [
          {
            id: "rsharma",
            name: "Rohit Sharma",
            isCaptain: true,
            image: "/players/rsharma.png",
          },
          {
            id: "ikishan",
            name: "Ishan Kishan",
            image: "/players/ikishan.png",
          },
          {
            id: "hpandya",
            name: "Hardik Pandya",
            image: "/players/hpandya.png",
          },
          {
            id: "kpollard",
            name: "Kieron Pollard",
            image: "/players/kpollard.png",
          },
          {
            id: "syadav",
            name: "Suryakumar Yadav",
            image: "/players/syadav.png",
          },
          {
            id: "jneesham",
            name: "Jimmy Neesham",
            image: "/players/jneesham.png",
          },
          {
            id: "kpandya",
            name: "Krunal Pandya",
            image: "/players/kpandya.png",
          },
          {
            id: "ncoultile",
            name: "Nathan Coulter-Nile",
            image: "/players/ncoultile.png",
          },
          {
            id: "pchawla",
            name: "Piyush Chawla",
            image: "/players/pchawla.png",
          },
          {
            id: "jbumrah",
            name: "Jasprit Bumrah",
            image: "/players/jbumrah.png",
          },
          {
            id: "tboult",
            name: "Trent Boult",
            image: "/players/tboult.png",
          },
        ],
        extras: {
          total: 0, // Placeholder as not provided in the data
          wides: 0,
          noBalls: 0,
          legByes: 0,
          byes: 0,
        },
      },
      score: 26, // Based on the first 2 overs data from the second document
      wickets: 0,
      overs: 2.0, // Completed 2 overs based on available data
      battingPerformances: [
        {
          playerId: "rsharma",
          runs: 1,
          balls: 4,
          fours: 0,
          sixes: 0,
          strikeRate: 25.0,
          isNotOut: true,
        },
        {
          playerId: "ikishan",
          runs: 25,
          balls: 8,
          fours: 4,
          sixes: 1,
          strikeRate: 312.5,
          isNotOut: true,
        },
      ],
      bowlingPerformances: [],
      fallOfWickets: [],
    },
    {
      team: {
        id: "srh",
        name: "Sunrisers Hyderabad",
        shortName: "SRH",
        logo: "/logos/srh.png",
        players: [
          {
            id: "jroy",
            name: "Jason Roy",
            image: "/players/jroy.png",
          },
          {
            id: "asharma",
            name: "Abhishek Sharma",
            image: "/players/asharma.png",
          },
          {
            id: "mpandey",
            name: "Manish Pandey",
            image: "/players/mpandey.png",
          },
          {
            id: "mnabi",
            name: "Mohammad Nabi",
            image: "/players/mnabi.png",
          },
          {
            id: "asamad",
            name: "Abdul Samad",
            image: "/players/asamad.png",
          },
          {
            id: "pgarg",
            name: "Priyam Garg",
            image: "/players/pgarg.png",
          },
          {
            id: "jholder",
            name: "Jason Holder",
            image: "/players/jholder.png",
          },
          {
            id: "rkhan",
            name: "Rashid Khan",
            image: "/players/rkhan.png",
          },
          {
            id: "wsaha",
            name: "Wriddhiman Saha",
            isWicketKeeper: true,
            image: "/players/wsaha.png",
          },
          {
            id: "skaul",
            name: "Siddarth Kaul",
            image: "/players/skaul.png",
          },
          {
            id: "umalik",
            name: "Umran Malik",
            image: "/players/umalik.png",
          },
        ],
        yetToBat: [],
      },
      score: 0, // Not provided in the data
      wickets: 0,
      overs: 0,
      battingPerformances: [],
      bowlingPerformances: [
        {
          playerId: "mnabi",
          overs: 1,
          maidens: 0,
          runs: 8,
          wickets: 0,
          economy: 8.0,
        },
        {
          playerId: "skaul",
          overs: 1,
          maidens: 0,
          runs: 18,
          wickets: 0,
          economy: 18.0,
        },
      ],
      fallOfWickets: [],
    },
  ],
  playerOfMatch: "Ishan Kishan",
  toss: {
    winner: "Mumbai Indians",
    decision: "bat",
  },
  officials: {
    umpires: ["Tapan Sharma", "VK Sharma"],
    thirdUmpire: "UV Gandhe", // Reserve umpire in the data
    matchReferee: "V Narayan Kutty",
    reserveUmpire: "UV Gandhe",
  },
};
