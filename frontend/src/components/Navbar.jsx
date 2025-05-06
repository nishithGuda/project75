import React from "react";
import { Link, useLocation } from "react-router-dom";

const Navbar = () => {
  const location = useLocation();
  const navItems = [
    { label: "Accounts", path: "/" },
    { label: "Transactions", path: "/transactions" },
    { label: "Transfer", path: "/transfer" },
  ];

  return (
    <nav className="bg-background py-4 px-6 flex items-center justify-between font-sans">
      <h1 className="text-2xl font-bold text-primary">MockUI for testing</h1>
      <ul className="flex space-x-12 text-xl font-medium">
        {navItems.map((item) => (
          <li key={item.path}>
            <Link
              to={item.path}
              data-id={`nav-${item.label.toLowerCase()}`}
              data-type="navigation"
              data-label={`${item.label} Navigation Link`}
              className={`${
                location.pathname === item.path
                  ? "text-accent border-b-2 border-accent"
                  : "text-text-secondary hover:text-accent"
              } pb-1 transition duration-150`}
            >
              {item.label}
            </Link>
          </li>
        ))}
      </ul>
    </nav>
  );
};

export default Navbar;
