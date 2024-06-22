import abc
from typing import Any, Dict

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy import Text as TextType
from sqlalchemy.engine import Compiled
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import DeclarativeMeta, declarative_base, relationship
from sqlalchemy.sql.expression import FunctionElement

Base: DeclarativeMeta = declarative_base()


class CustomOperations:
    __table__ = abc.abstractproperty()

    def to_dict(self) -> Dict:
        d = {}
        for column in self.__table__.columns:
            d[column.name] = getattr(self, column.name)
        d["__tablename__"] = self.__class__
        return d

    def update(self, **kwargs: Any) -> None:
        column_names = [column.name for column in self.__table__.columns]
        for attr in kwargs:
            if attr in column_names:
                setattr(self, attr, kwargs[attr])


class utcnow(FunctionElement):
    type = DateTime()
    inherit_cache = True


@compiles(utcnow, "postgresql")
def pg_utcnow(element: FunctionElement, compiler: Compiled, **kw: Any) -> str:
    return "TIMEZONE('utc', CURRENT_TIMESTAMP)"


@compiles(utcnow, "sqlite")
def sqlite_utcnow(element: FunctionElement, compiler: Compiled, **kw: Any) -> str:
    return "DATETIME('now')"


class CookieBanner(Base, CustomOperations):
    __tablename__ = "cookie_banners"
    id = Column(Integer, primary_key=True)

    website_id = Column(Integer, ForeignKey("websites.id"))
    website = relationship(
        "Website", back_populates="cookie_banners", uselist=False, lazy="subquery"
    )

    detected = Column(Integer, default=0)
    detection_method = Column(String, default="none")
    cookie_mention = Column(Integer, default=0)
    cookie_mention_not_in_link = Column(Integer, default=0)
    html = Column(TextType, default="")
    text = Column(TextType, default="")
    width = Column(Integer, default=0)
    height = Column(Integer, default=0)
    position_x = Column(Integer, default=0)
    position_y = Column(Integer, default=0)
    z_index = Column(Integer)
    selector = Column(TextType, default="")
    dfs_max_depth = Column(Integer, default=0)
    link_to_text_ratio = Column(Float, default=0.0)
    timestamp = Column(DateTime, server_default=utcnow())


class CookieTimestamp(Base, CustomOperations):
    __tablename__ = "cookie_timestamps"
    id = Column(Integer, primary_key=True)
    website_id = Column(Integer, ForeignKey("websites.id"))
    website = relationship(
        "Website", back_populates="cookie_timestamps", uselist=False, lazy="subquery"
    )
    visit_id = Column(BigInteger, nullable=False)
    collection_strategy = Column(String, nullable=False)
    start_timestamp = Column(DateTime, nullable=False)
    end_timestamp = Column(DateTime, nullable=False)
    num_cookies = Column(Integer, nullable=False)
    click_timestamp = Column(DateTime, nullable=True)


class Text(Base, CustomOperations):
    __tablename__ = "cb_text"
    id = Column(Integer, primary_key=True)
    website_id = Column(Integer, ForeignKey("websites.id"))
    website = relationship(
        "Website", back_populates="cb_text", uselist=False, lazy="subquery"
    )
    num_clicks = Column(Integer, nullable=False)
    text = Column(TextType, nullable=False)
    timestamp = Column(DateTime, server_default=utcnow())

    purpose_predictions = relationship(
        "PurposePrediction", back_populates="sentence", uselist=False
    )
    purpose_annotations = relationship(
        "PurposeAnnotation", back_populates="sentence", uselist=False
    )


class InteractiveElementText(Base, CustomOperations):
    __tablename__ = "interactive_elements_text"
    id = Column(Integer, primary_key=True)
    website_id = Column(Integer, ForeignKey("websites.id"))
    website = relationship(
        "Website",
        back_populates="interactive_elements_text",
        uselist=False,
        lazy="subquery",
    )
    num_clicks = Column(Integer, nullable=False)
    text = Column(TextType, nullable=False)
    selector = Column(TextType, nullable=False)
    timestamp = Column(DateTime, server_default=utcnow())


class NumInteractiveElements(Base, CustomOperations):
    __tablename__ = "num_interactive_elements"
    id = Column(Integer, primary_key=True)
    website_id = Column(Integer, ForeignKey("websites.id"))
    website = relationship(
        "Website",
        back_populates="num_interactive_elements",
        uselist=False,
        lazy="subquery",
    )
    depth = Column(Integer, nullable=False)
    num_elements = Column(Integer, nullable=False)
    timestamp = Column(DateTime, server_default=utcnow())


class Link(Base, CustomOperations):
    __tablename__ = "cb_links"
    id = Column(Integer, primary_key=True)
    website_id = Column(Integer, ForeignKey("websites.id"))
    website = relationship(
        "Website", back_populates="cb_links", uselist=False, lazy="subquery"
    )
    num_clicks = Column(Integer, nullable=False)
    text = Column(TextType, nullable=False)
    timestamp = Column(DateTime, server_default=utcnow())


class PurposePrediction(Base, CustomOperations):
    __tablename__ = "purpose_predictions"
    id = Column(Integer, primary_key=True)
    sentence_id = Column(Integer, ForeignKey("cb_text.id"))
    sentence = relationship(
        "Text", back_populates="purpose_predictions", uselist=False, lazy="subquery"
    )
    purpose_detected = Column(Integer, nullable=False)
    purpose_classification = Column(TextType, nullable=True)


class PurposeAnnotation(Base, CustomOperations):
    __tablename__ = "purpose_annotations"
    id = Column(Integer, primary_key=True)
    sentence_id = Column(Integer, ForeignKey("cb_text.id"))
    sentence = relationship(
        "Text", back_populates="purpose_annotations", uselist=False, lazy="subquery"
    )
    purpose_detected = Column(Integer, nullable=False)
    purpose_classification = Column(TextType, nullable=True)


class Errors(Base, CustomOperations):
    __tablename__ = "errors"
    id = Column(Integer, primary_key=True)
    website_id = Column(Integer, ForeignKey("websites.id"))
    website = relationship(
        "Website", back_populates="errors", uselist=False, lazy="subquery"
    )
    text = Column(String, nullable=False)
    timestamp = Column(DateTime, server_default=utcnow())


class Cookies(Base, CustomOperations):
    __tablename__ = "javascript_cookies"
    id = Column(Integer, primary_key=True)
    browser_id = Column(BigInteger, nullable=False)
    visit_id = Column(BigInteger, nullable=False)
    extension_session_uuid = Column(String, nullable=False)
    event_ordinal = Column(Integer, nullable=False)
    record_type = Column(String, nullable=False)
    change_cause = Column(String, nullable=False)
    expiry = Column(DateTime, nullable=False)
    is_http_only = Column(Integer, nullable=False)
    is_host_only = Column(Integer, nullable=False)
    is_session = Column(Integer, nullable=False)
    cookie_domain = Column(String, nullable=False)
    is_secure = Column(Integer, nullable=False)
    name = Column(String, nullable=False)
    path = Column(String, nullable=False)
    value = Column(TextType, nullable=False)
    same_site = Column(String, nullable=False)
    website_id = Column(Integer, ForeignKey("websites.id"))
    website = relationship(
        "Website", back_populates="cookies", uselist=False, lazy="subquery"
    )
    collection_strategy = Column(String, nullable=False)
    store_id = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    predicted = Column(Integer, nullable=True)


class CookiesWithPredictions(Base, CustomOperations):
    __tablename__ = "cookies_with_predictions"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    cookie_domain = Column(String, nullable=False)
    visit_id = Column(BigInteger, nullable=False)
    website_id = Column(Integer, ForeignKey("websites.id"))
    website = relationship(
        "Website",
        back_populates="cookies_with_predictions",
        uselist=False,
        lazy="subquery",
    )
    path = Column(String, nullable=False)
    collection_strategy = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    classification = Column(Integer, nullable=True)


class ConsentOption(Base, CustomOperations):
    __tablename__ = "consent_options"
    id = Column(Integer, primary_key=True)
    website_id = Column(Integer, ForeignKey("websites.id"))
    website = relationship(
        "Website", back_populates="consent_options", uselist=False, lazy="subquery"
    )
    consent_option = Column(String, nullable=False)
    text = Column(String, nullable=False)
    selector = Column(String, nullable=False)
    iframe = Column(String, nullable=False)
    timestamp = Column(DateTime, server_default=utcnow())


class CrawlResults(Base, CustomOperations):
    __tablename__ = "crawl_results"
    id = Column(Integer, primary_key=True)
    website_id = Column(Integer, ForeignKey("websites.id"))
    website = relationship(
        "Website", back_populates="crawl_results", uselist=False, lazy="subquery"
    )
    timestamp = Column(DateTime, server_default=utcnow())
    # while crawling
    cookie_notice_detected = Column(Boolean)
    accept_button_detected = Column(Integer)
    reject_button_detected = Column(Integer)
    close_button_detected = Column(Integer)
    save_button_detected = Column(Integer)
    accept_button_detected_without_reject_button = Column(Boolean)
    cmp_detected = Column(Boolean)
    mentions_legitimate_interest_in_initial_text = Column(Boolean)
    mentions_legitimate_interest = Column(Boolean)
    interaction_depth = Column(Integer, default=0)
    # dark patterns, while crawling
    forced_action_detected = Column(Boolean)
    nagging_detected = Column(Boolean)
    interface_interference_detected = Column(Boolean)
    interface_interference_analysis = Column(JSON)
    obstruction_detected = Column(Boolean)
    # after CookieBlock predictions
    tracking_detected = Column(Integer)
    tracking_detected_after_reject = Column(Integer)
    tracking_detected_after_close = Column(Integer)
    tracking_detected_after_save = Column(Integer)
    tracking_detected_prior_to_interaction = Column(Integer)
    # after text predictions
    tracking_purposes_detected_in_initial_text = Column(Integer)
    tracking_purposes_detected = Column(Integer)
    other = Column(JSON, nullable=True)


class ExpectedResults(Base, CustomOperations):
    __tablename__ = "expected_results"
    id = Column(Integer, primary_key=True)
    website_id = Column(Integer, ForeignKey("websites.id"))
    website = relationship(
        "Website", back_populates="expected_results", uselist=False, lazy="subquery"
    )
    timestamp = Column(DateTime, server_default=utcnow())
    # while crawling
    cookie_notice_detected = Column(Boolean)
    accept_button_detected = Column(Integer)
    reject_button_detected = Column(Integer)
    close_button_detected = Column(Integer)
    save_button_detected = Column(Integer)
    accept_button_detected_without_reject_button = Column(Boolean)
    cmp_detected = Column(Boolean)
    mentions_legitimate_interest_in_initial_text = Column(Boolean)
    mentions_legitimate_interest = Column(Boolean)
    interaction_depth = Column(Integer)
    # dark patterns, while crawling
    forced_action_detected = Column(Boolean)
    nagging_detected = Column(Boolean)
    interface_interference_detected = Column(Boolean)
    interface_interference_analysis = Column(JSON)
    obstruction_detected = Column(Boolean)
    # after CookieBlock predictions
    tracking_detected = Column(Integer)
    tracking_detected_after_reject = Column(Integer)
    tracking_detected_after_close = Column(Integer)
    tracking_detected_after_save = Column(Integer)
    tracking_detected_prior_to_interaction = Column(Integer)
    # after text predictions
    tracking_purposes_detected_in_initial_text = Column(Integer)
    tracking_purposes_detected = Column(Integer)
    other = Column(JSON, nullable=True)


class Website(Base, CustomOperations):
    __tablename__ = "websites"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    url = Column(String, nullable=False)
    save_path = Column(String, nullable=False)
    timestamp = Column(DateTime, server_default=utcnow())
    visit_id = Column(BigInteger)
    crux_country = Column(String, nullable=False)
    crux_rank = Column(Integer, nullable=False)
    language = Column(String)
    success = Column(Integer)
    previously_timed_out = Column(Integer)
    gpc_detected = Column(Integer)
    similarweb_data = Column(JSON)
    ranking_data = Column(JSON)
    cmp = Column(JSON)
    other = Column(JSON)

    experiment_id = Column(String, ForeignKey("experiments.id"))
    experiment = relationship(
        "Experiment", back_populates="websites", uselist=False, lazy="subquery"
    )

    cookie_banners = relationship(
        "CookieBanner", back_populates="website", cascade="all, delete", uselist=False
    )
    cookie_timestamps = relationship(
        "CookieTimestamp", back_populates="website", cascade="all, delete", uselist=True
    )
    cookies = relationship(
        "Cookies", back_populates="website", cascade="all, delete", uselist=True
    )
    cookies_with_predictions = relationship(
        "CookiesWithPredictions",
        back_populates="website",
        cascade="all, delete",
        uselist=True,
    )
    errors = relationship(
        "Errors", back_populates="website", cascade="all, delete", uselist=True
    )

    consent_options = relationship(
        "ConsentOption", back_populates="website", cascade="all, delete", uselist=True
    )
    cb_text = relationship(
        "Text", back_populates="website", cascade="all, delete", uselist=True
    )
    cb_links = relationship(
        "Link", back_populates="website", cascade="all, delete", uselist=True
    )
    interactive_elements_text = relationship(
        "InteractiveElementText",
        back_populates="website",
        cascade="all, delete",
        uselist=True,
    )
    num_interactive_elements = relationship(
        "NumInteractiveElements",
        back_populates="website",
        cascade="all, delete",
        uselist=True,
    )
    crawl_results = relationship(
        "CrawlResults", back_populates="website", cascade="all, delete", uselist=False
    )
    execution_times = relationship(
        "ExecutionTimes", back_populates="website", cascade="all, delete", uselist=False
    )

    expected_results = relationship(
        "ExpectedResults",
        back_populates="website",
        cascade="all, delete",
        uselist=False,
    )


class ExecutionTimes(Base, CustomOperations):
    __tablename__ = "execution_times"
    id = Column(Integer, primary_key=True)
    website_id = Column(Integer, ForeignKey("websites.id"))
    website = relationship(
        "Website", back_populates="execution_times", uselist=False, lazy="subquery"
    )
    notice_detection = Column(Float, default=0.0)
    initial_cookie_extraction = Column(Float, default=0.0)
    exploration_naive = Column(Float, default=0.0)
    exploration_with_ietc_model = Column(Float, default=0.0)


class Experiment(Base, CustomOperations):
    __tablename__ = "experiments"
    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, server_default=utcnow())
    num_full_iterations = Column(Integer, default=0)
    country = Column(String)
    region = Column(String)
    config = Column(JSON)

    websites = relationship(
        "Website", back_populates="experiment", cascade="all, delete", uselist=True
    )
